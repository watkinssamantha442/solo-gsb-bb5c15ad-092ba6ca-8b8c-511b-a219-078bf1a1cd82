"""Startup migration chain tests (PRAGMA user_version).

Covers the requirements of the refactored SQLite startup migration:

- a brand-new database is built and stamped at LATEST_USER_VERSION in one go;
- a version-0 database from an older release traverses the ordered chain and
  keeps every row, index value and constraint behavior;
- a version-0 database already carrying the final schema (last-release file)
  runs idempotently through the guarded steps;
- each step's DDL and its user_version bump commit together: an injected
  step failure rolls the step back, leaves the previous whole version in
  place, and the chain resumes after the "repair" on the next start;
- several instances opening one legacy file concurrently are serialized by
  the write lock: each step is applied exactly once and every instance ends
  at the latest version;
- a foreign-key integrity violation after migration aborts construction
  (so the application cannot start workers against an inconsistent file);
- a database stamped by a newer release is rejected;
- the FastAPI lifespan propagates the migration failure and never reaches
  worker creation.

Pure stdlib + the app's own modules; run with either of

    python3 -m unittest tests.test_db_migrations -v
    python3 -m pytest tests/test_db_migrations.py
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
import unittest
from dataclasses import replace
from unittest.mock import patch

from app import config as app_config
from app import db as dbmod
from app.db import (
    Database,
    LATEST_USER_VERSION,
    Migration,
    MigrationError,
    _FRESH_SCHEMA_STATEMENTS,
    _V1_BOOTSTRAP_STATEMENTS,
)


def _raw_connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA foreign_keys=OFF")
    return conn


def _create_v1_database(path: str) -> None:
    """Create a database exactly as the original release left it: v1 tables,
    user_version 0, and representative data in every table."""
    conn = _raw_connect(path)
    try:
        for stmt in _V1_BOOTSTRAP_STATEMENTS:
            conn.execute(stmt)
        conn.execute(
            "INSERT INTO subscriptions (id, target_url, secret, created_at) "
            "VALUES ('sub-aaaa0001', 'http://example/hook', 's3cr3t', 100.0)"
        )
        conn.execute(
            "INSERT INTO events (id, payload, created_at) VALUES ('evt-1', '{}', 101.0)"
        )
        conn.execute(
            """INSERT INTO tasks (event_id, subscription_id, status, attempts,
                  max_attempts, next_attempt_at, claimed_by, last_status_code,
                  last_error, result, created_at, updated_at)
               VALUES ('evt-1', 'sub-aaaa0001', 'failed', 4, 4, NULL, 'w1', 500,
                  'boom', 'failed', 102.0, 109.0)"""
        )
        conn.execute(
            """INSERT INTO attempts (task_id, attempt_no, status_code, error,
                  started_at, finished_at)
               VALUES (1, 1, 500, 'boom', 102.0, 103.0)"""
        )
        conn.execute("PRAGMA user_version=0")
    finally:
        conn.close()


def _create_drifted_cancelled_column_database(path: str) -> None:
    """Create the half-finished version-0 database: tasks has the
    cancelled_at column (a drifting release ALTERed it in) but its status
    CHECK still lists only the original four states, so cancelling a task
    raises a CHECK violation. This drifting release also carries deliver_at
    with real scheduled data. Includes representative rows with an
    AUTOINCREMENT id gap, attempts history, a pending task and a non-NULL
    cancelled_at value."""
    conn = _raw_connect(path)
    try:
        conn.execute(
            "CREATE TABLE subscriptions (id TEXT PRIMARY KEY, target_url TEXT "
            "NOT NULL, secret TEXT NOT NULL, created_at REAL NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE events (id TEXT PRIMARY KEY, payload TEXT NOT NULL, "
            "created_at REAL NOT NULL)"
        )
        conn.execute(
            """CREATE TABLE tasks (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 event_id TEXT NOT NULL REFERENCES events(id),
                 subscription_id TEXT NOT NULL,
                 status TEXT NOT NULL DEFAULT 'pending'
                   CHECK (status IN ('pending', 'in_progress', 'delivered', 'failed')),
                 attempts INTEGER NOT NULL DEFAULT 0,
                 max_attempts INTEGER NOT NULL,
                 deliver_at REAL,
                 next_attempt_at REAL, claimed_by TEXT,
                 last_status_code INTEGER, last_error TEXT,
                 result TEXT CHECK (result IN ('delivered', 'failed')),
                 created_at REAL NOT NULL, updated_at REAL NOT NULL,
                 cancelled_at REAL,
                 UNIQUE (event_id, subscription_id))"""
        )
        conn.execute(
            "CREATE TABLE attempts (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "task_id INTEGER NOT NULL REFERENCES tasks(id), attempt_no INTEGER "
            "NOT NULL, status_code INTEGER, error TEXT, started_at REAL NOT NULL, "
            "finished_at REAL NOT NULL)"
        )
        conn.execute("CREATE INDEX idx_tasks_due ON tasks (status, next_attempt_at)")
        conn.execute("CREATE INDEX idx_tasks_sub_order ON tasks (subscription_id, id, status)")
        conn.execute("CREATE INDEX idx_attempts_task ON attempts (task_id)")
        conn.execute(
            "INSERT INTO subscriptions (id, target_url, secret, created_at) "
            "VALUES ('sub-dddd0004', 'http://example/4', 'k4', 1.0)"
        )
        conn.execute(
            "INSERT INTO subscriptions (id, target_url, secret, created_at) "
            "VALUES ('sub-dddd0005', 'http://example/5', 'k5', 1.0)"
        )
        conn.execute(
            "INSERT INTO subscriptions (id, target_url, secret, created_at) "
            "VALUES ('sub-dddd0006', 'http://example/6', 'k6', 1.0)"
        )
        conn.execute(
            "INSERT INTO subscriptions (id, target_url, secret, created_at) "
            "VALUES ('sub-dddd0007', 'http://example/7', 'k7', 1.0)"
        )
        conn.execute(
            "INSERT INTO events (id, payload, created_at) "
            "VALUES ('evt-drift', '{}', 1.0)"
        )
        # Rows with an AUTOINCREMENT gap (ids 1, 2 then 11) so the
        # high-water mark in sqlite_sequence is clearly verifiable.
        conn.execute(
            "INSERT INTO tasks (id, event_id, subscription_id, status, attempts, "
            "max_attempts, next_attempt_at, created_at, updated_at) VALUES "
            "(1, 'evt-drift', 'sub-dddd0004', 'delivered', 1, 4, NULL, 1.0, 2.0)"
        )
        conn.execute(
            "INSERT INTO tasks (id, event_id, subscription_id, status, attempts, "
            "max_attempts, next_attempt_at, created_at, updated_at) VALUES "
            "(2, 'evt-drift', 'sub-dddd0005', 'failed', 4, 4, NULL, 1.0, 3.0)"
        )
        conn.execute(
            "INSERT INTO tasks (id, event_id, subscription_id, status, attempts, "
            "max_attempts, deliver_at, next_attempt_at, created_at, updated_at) VALUES "
            "(11, 'evt-drift', 'sub-dddd0006', 'pending', 0, 4, 9.9e99, 9.9e99, 4.0, 4.0)"
        )
        # Non-NULL cancelled_at on a drifting table (illegal-ish state that
        # must survive the rebuild byte-for-byte), plus a deliver_at value
        # the rebuild must carry over instead of replacing it with NULL.
        conn.execute(
            "INSERT INTO events (id, payload, created_at) "
            "VALUES ('evt-other', '{}', 1.0)"
        )
        conn.execute(
            "INSERT INTO tasks (id, event_id, subscription_id, status, attempts, "
            "max_attempts, deliver_at, created_at, updated_at, cancelled_at) VALUES "
            "(12, 'evt-other', 'sub-dddd0007', 'failed', 1, 4, 55.5, 1.0, 5.0, 7.75)"
        )
        conn.execute(
            "INSERT INTO attempts (task_id, attempt_no, status_code, error, "
            "started_at, finished_at) VALUES "
            "(1, 1, 200, NULL, 1.0, 2.0), "
            "(2, 1, 500, 'boom', 1.0, 1.5), "
            "(2, 2, 500, 'boom', 1.5, 2.0)"
        )
        conn.execute("PRAGMA user_version=0")
    finally:
        conn.close()


class MigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="wr-migrate-")
        self.path = os.path.join(self.tmp, "test.db")

    def tearDown(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(self.path + suffix)
            except FileNotFoundError:
                pass
        os.rmdir(self.tmp)

    # ------------------------------------------------------------------
    # fresh database
    # ------------------------------------------------------------------
    def test_fresh_database_is_stamped_at_latest_version(self) -> None:
        database = Database(self.path)
        try:
            conn = database._conn
            self.assertEqual(
                conn.execute("PRAGMA user_version").fetchone()[0],
                LATEST_USER_VERSION,
            )
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            self.assertEqual(
                tables,
                {
                    "subscriptions", "events", "tasks", "attempts",
                    "ingested_events", "subscription_topics", "topic_events",
                    "sqlite_sequence",
                },
            )
            indexes = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
                )
            }
            self.assertEqual(
                indexes,
                {
                    "idx_tasks_due", "idx_tasks_sub_order",
                    "idx_attempts_task", "idx_subscription_topics_topic",
                },
            )
            # foreign-key enforcement is enabled for normal operation.
            self.assertEqual(
                conn.execute("PRAGMA foreign_keys").fetchone()[0], 1
            )
        finally:
            database.close()

    def test_fresh_database_reopen_is_a_noop(self) -> None:
        database = Database(self.path)
        database.close()
        reopened = Database(self.path)
        try:
            self.assertEqual(
                reopened._conn.execute("PRAGMA user_version").fetchone()[0],
                LATEST_USER_VERSION,
            )
        finally:
            reopened.close()

    # ------------------------------------------------------------------
    # legacy databases traverse the chain
    # ------------------------------------------------------------------
    def test_v1_database_is_upgraded_with_all_data_preserved(self) -> None:
        _create_v1_database(self.path)
        database = Database(self.path)
        try:
            conn = database._conn
            self.assertEqual(
                conn.execute("PRAGMA user_version").fetchone()[0],
                LATEST_USER_VERSION,
            )
            # Subscription row and its ALTER-added columns survive.
            sub = conn.execute(
                "SELECT target_url, secret, paused_at, breaker_state, "
                "breaker_consecutive_failures, rate_capacity, window_timezone, "
                "window_updated_at FROM subscriptions WHERE id = 'sub-aaaa0001'"
            ).fetchone()
            self.assertEqual(sub[0], "http://example/hook")
            self.assertEqual(sub[1], "s3cr3t")
            self.assertIsNone(sub[2])
            self.assertEqual(sub[3], "closed")
            self.assertEqual(sub[4], 0)
            self.assertIsNone(sub[5])
            self.assertIsNone(sub[6])
            self.assertIsNone(sub[7])

            # Existing task row copied through the table rebuild.
            task = conn.execute(
                "SELECT status, attempts, result, cancelled_at, deliver_at "
                "FROM tasks WHERE id = 1"
            ).fetchone()
            self.assertEqual(task[0], "failed")
            self.assertEqual(task[1], 4)
            self.assertEqual(task[2], "failed")
            self.assertIsNone(task[3])
            self.assertIsNone(task[4])

            # Attempt history preserved.
            attempt = conn.execute(
                "SELECT attempt_no, status_code FROM attempts WHERE task_id = 1"
            ).fetchone()
            self.assertEqual(tuple(attempt), (1, 500))

            # The rebuilt CHECK accepts the 'cancelled' terminal state, and
            # the recreated indexes are present and functional.
            conn.execute(
                "INSERT INTO events (id, payload, created_at) "
                "VALUES ('evt-2', '{}', 110.0)"
            )
            conn.execute(
                """INSERT INTO tasks (event_id, subscription_id, status,
                      attempts, max_attempts, next_attempt_at, created_at,
                      updated_at, cancelled_at)
                   VALUES ('evt-2', 'sub-aaaa0001', 'cancelled', 0, 4, NULL,
                      110.0, 110.0, 110.0)"""
            )
            self.assertEqual(
                [
                    tuple(r)
                    for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='index' "
                        "AND tbl_name='tasks' AND name NOT LIKE 'sqlite_%' "
                        "ORDER BY name"
                    )
                ],
                [("idx_tasks_due",), ("idx_tasks_sub_order",)],
            )
        finally:
            database.close()

    def test_v1_database_can_be_reopened_after_upgrade(self) -> None:
        _create_v1_database(self.path)
        Database(self.path).close()
        reopened = Database(self.path)
        try:
            self.assertEqual(
                reopened._conn.execute("PRAGMA user_version").fetchone()[0],
                LATEST_USER_VERSION,
            )
            self.assertEqual(
                reopened._conn.execute(
                    "SELECT COUNT(*) FROM subscriptions"
                ).fetchone()[0],
                1,
            )
        finally:
            reopened.close()

    def test_version0_with_final_schema_is_idempotently_stamped(self) -> None:
        # A database left by the previous release: final schema in place but
        # user_version still 0 (the old migrator never stamped a version).
        conn = _raw_connect(self.path)
        for stmt in _FRESH_SCHEMA_STATEMENTS:
            conn.execute(stmt)
        conn.execute(
            "INSERT INTO subscriptions (id, target_url, secret, created_at) "
            "VALUES ('sub-bbbb0002', 'http://example/2', 'k', 200.0)"
        )
        conn.execute("PRAGMA user_version=0")
        conn.close()

        database = Database(self.path)
        try:
            self.assertEqual(
                database._conn.execute("PRAGMA user_version").fetchone()[0],
                LATEST_USER_VERSION,
            )
            self.assertEqual(
                database._conn.execute(
                    "SELECT target_url FROM subscriptions WHERE id = 'sub-bbbb0002'"
                ).fetchone()[0],
                "http://example/2",
            )
        finally:
            database.close()

    def test_version0_with_cancelled_but_without_deliver_at(self) -> None:
        # Drift point: a release that had cancellation (widened CHECK and
        # cancelled_at) but predates scheduled delivery (no deliver_at).
        conn = _raw_connect(self.path)
        conn.execute(
            "CREATE TABLE subscriptions (id TEXT PRIMARY KEY, target_url TEXT "
            "NOT NULL, secret TEXT NOT NULL, created_at REAL NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE events (id TEXT PRIMARY KEY, payload TEXT NOT NULL, "
            "created_at REAL NOT NULL)"
        )
        conn.execute(
            """CREATE TABLE tasks (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 event_id TEXT NOT NULL REFERENCES events(id),
                 subscription_id TEXT NOT NULL,
                 status TEXT NOT NULL DEFAULT 'pending'
                   CHECK (status IN ('pending','in_progress','delivered',
                                     'failed','cancelled')),
                 attempts INTEGER NOT NULL DEFAULT 0,
                 max_attempts INTEGER NOT NULL,
                 next_attempt_at REAL, claimed_by TEXT,
                 last_status_code INTEGER, last_error TEXT,
                 result TEXT CHECK (result IN ('delivered','failed')),
                 created_at REAL NOT NULL, updated_at REAL NOT NULL,
                 cancelled_at REAL,
                 UNIQUE (event_id, subscription_id))"""
        )
        conn.execute(
            "CREATE TABLE attempts (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "task_id INTEGER NOT NULL REFERENCES tasks(id), attempt_no INTEGER "
            "NOT NULL, status_code INTEGER, error TEXT, started_at REAL NOT NULL, "
            "finished_at REAL NOT NULL)"
        )
        conn.execute(
            "INSERT INTO subscriptions (id, target_url, secret, created_at) "
            "VALUES ('sub-cccc0003', 'u', 'k', 1.0)"
        )
        conn.execute(
            "INSERT INTO events (id, payload, created_at) VALUES ('e', '{}', 1.0)"
        )
        conn.execute(
            "INSERT INTO tasks (event_id, subscription_id, status, attempts, "
            "max_attempts, created_at, updated_at, cancelled_at) VALUES "
            "('e', 'sub-cccc0003', 'cancelled', 0, 4, 1.0, 1.0, 1.0)"
        )
        conn.close()

        database = Database(self.path)
        try:
            cols = {
                r[1]
                for r in database._conn.execute("PRAGMA table_info(tasks)")
            }
            self.assertIn("deliver_at", cols)
            self.assertIn("cancelled_at", cols)
            self.assertEqual(
                database._conn.execute("PRAGMA user_version").fetchone()[0],
                LATEST_USER_VERSION,
            )
            # The pre-existing cancelled row is intact.
            row = database._conn.execute(
                "SELECT status, deliver_at FROM tasks WHERE id = 1"
            ).fetchone()
            self.assertEqual(tuple(row), ("cancelled", None))
        finally:
            database.close()

    # ------------------------------------------------------------------
    # drifted version-0 file: cancelled_at column present but the status
    # CHECK still rejects 'cancelled' (half-finished older migration)
    # ------------------------------------------------------------------
    def test_drifted_column_without_check_is_rebuilt_with_data_intact(self) -> None:
        _create_drifted_cancelled_column_database(self.path)

        # Precondition: the drifting file rejects a 'cancelled' status even
        # though the column exists, so column-presence alone cannot gate v7.
        raw = _raw_connect(self.path)
        try:
            self.assertEqual(raw.execute("PRAGMA user_version").fetchone()[0], 0)
            cols = {r[1] for r in raw.execute("PRAGMA table_info(tasks)")}
            self.assertIn("cancelled_at", cols)
            with self.assertRaises(sqlite3.IntegrityError):
                raw.execute(
                    "INSERT INTO events (id, payload, created_at) "
                    "VALUES ('evt-pre', '{}', 1.0)"
                )
                raw.execute(
                    "INSERT INTO tasks (event_id, subscription_id, status, "
                    "max_attempts, created_at, updated_at) VALUES "
                    "('evt-pre', 'sub-dddd0004', 'cancelled', 4, 1, 1)"
                )
        finally:
            raw.close()

        database = Database(self.path)
        try:
            conn = database._conn
            self.assertEqual(
                conn.execute("PRAGMA user_version").fetchone()[0],
                LATEST_USER_VERSION,
            )

            # Every original task row survives with its primary key, values
            # and the drifting table's non-NULL cancelled_at.
            rows = conn.execute(
                "SELECT id, status, attempts, cancelled_at, deliver_at, "
                "next_attempt_at FROM tasks ORDER BY id"
            ).fetchall()
            self.assertEqual(
                [tuple(r) for r in rows],
                [
                    (1, "delivered", 1, None, None, None),
                    (2, "failed", 4, None, None, None),
                    (11, "pending", 0, None, 9.9e99, 9.9e99),
                    (12, "failed", 1, 7.75, 55.5, None),
                ],
            )

            # Attempt history and its foreign key to the rebuilt tasks table.
            attempts = conn.execute(
                "SELECT task_id, attempt_no, status_code FROM attempts "
                "ORDER BY id"
            ).fetchall()
            self.assertEqual(
                [tuple(r) for r in attempts],
                [(1, 1, 200), (2, 1, 500), (2, 2, 500)],
            )
            self.assertEqual(
                conn.execute("PRAGMA foreign_key_check").fetchall(), []
            )

            # The rebuilt table truly enforces cancellation now.
            conn.execute(
                "UPDATE tasks SET status='cancelled', cancelled_at=123.0 "
                "WHERE id = 11"
            )
            self.assertEqual(
                conn.execute(
                    "SELECT status FROM tasks WHERE id = 11"
                ).fetchone()[0],
                "cancelled",
            )
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO events (id, payload, created_at) "
                    "VALUES ('evt-bogus', '{}', 1.0)"
                )
                conn.execute(
                    "INSERT INTO tasks (event_id, subscription_id, status, "
                    "max_attempts, created_at, updated_at) VALUES "
                    "('evt-bogus', 'sub-dddd0004', 'bogus', 4, 1, 1)"
                )

            # Indexes are back.
            self.assertEqual(
                [
                    r[0]
                    for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='index' "
                        "AND tbl_name='tasks' AND name NOT LIKE 'sqlite_%' "
                        "ORDER BY name"
                    )
                ],
                ["idx_tasks_due", "idx_tasks_sub_order"],
            )

            # AUTOINCREMENT high-water mark carried over: the next insert
            # continues above the old max id (12), never reusing a freed id.
            conn.execute(
                "INSERT INTO events (id, payload, created_at) "
                "VALUES ('evt-seq', '{}', 1.0)"
            )
            cur = conn.execute(
                "INSERT INTO tasks (event_id, subscription_id, status, "
                "max_attempts, created_at, updated_at) VALUES "
                "('evt-seq', 'sub-dddd0004', 'pending', 4, 9, 9)"
            )
            self.assertGreater(cur.lastrowid, 12)
            self.assertEqual(
                conn.execute(
                    "SELECT seq FROM sqlite_sequence WHERE name='tasks'"
                ).fetchone()[0],
                cur.lastrowid,
            )
        finally:
            database.close()

    def test_drifted_column_no_deliver_at_without_check_is_rebuilt(self) -> None:
        # Same half-finished drift, but the source table also lacks
        # deliver_at entirely (the column list is derived from the source,
        # so the rebuild fills deliver_at with NULL and v8 then finds it
        # already present).
        conn = _raw_connect(self.path)
        try:
            conn.execute(
                "CREATE TABLE subscriptions (id TEXT PRIMARY KEY, target_url "
                "TEXT NOT NULL, secret TEXT NOT NULL, created_at REAL NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE events (id TEXT PRIMARY KEY, payload TEXT NOT "
                "NULL, created_at REAL NOT NULL)"
            )
            conn.execute(
                """CREATE TABLE tasks (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     event_id TEXT NOT NULL REFERENCES events(id),
                     subscription_id TEXT NOT NULL,
                     status TEXT NOT NULL DEFAULT 'pending'
                       CHECK (status IN ('pending','in_progress','delivered','failed')),
                     attempts INTEGER NOT NULL DEFAULT 0,
                     max_attempts INTEGER NOT NULL,
                     next_attempt_at REAL, claimed_by TEXT,
                     last_status_code INTEGER, last_error TEXT,
                     result TEXT CHECK (result IN ('delivered','failed')),
                     created_at REAL NOT NULL, updated_at REAL NOT NULL,
                     cancelled_at REAL,
                     UNIQUE (event_id, subscription_id))"""
            )
            conn.execute(
                "CREATE TABLE attempts (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "task_id INTEGER NOT NULL REFERENCES tasks(id), attempt_no "
                "INTEGER NOT NULL, status_code INTEGER, error TEXT, "
                "started_at REAL NOT NULL, finished_at REAL NOT NULL)"
            )
            conn.execute(
                "INSERT INTO subscriptions (id, target_url, secret, created_at) "
                "VALUES ('s1', 'u', 'k', 1)"
            )
            conn.execute(
                "INSERT INTO events (id, payload, created_at) "
                "VALUES ('e1', '{}', 1)"
            )
            conn.execute(
                "INSERT INTO tasks (event_id, subscription_id, status, "
                "max_attempts, created_at, updated_at, cancelled_at) "
                "VALUES ('e1', 's1', 'failed', 4, 1, 2, 3.5)"
            )
        finally:
            conn.close()

        database = Database(self.path)
        try:
            self.assertEqual(
                database._conn.execute("PRAGMA user_version").fetchone()[0],
                LATEST_USER_VERSION,
            )
            row = database._conn.execute(
                "SELECT status, deliver_at, cancelled_at FROM tasks WHERE id = 1"
            ).fetchone()
            self.assertEqual(tuple(row), ("failed", None, 3.5))
            result = database.cancel_event("e1")
            self.assertEqual(result["cancelled"], 0)  # no pending tasks
        finally:
            database.close()

    def test_drifted_database_cancel_endpoint_flips_pending_tasks(self) -> None:
        # After the rebuild, POST /events/{event_id}/cancel must work end to
        # end: pending tasks become 'cancelled' and the call is idempotent.
        _create_drifted_cancelled_column_database(self.path)
        database = Database(self.path)
        try:
            result = database.cancel_event("evt-drift")
            self.assertIsNotNone(result)
            self.assertEqual(result["cancelled"], 1)
            self.assertEqual(
                result["counts"],
                {"pending": 0, "in_progress": 0, "delivered": 1,
                 "failed": 1, "cancelled": 1},
            )
            row = database._conn.execute(
                "SELECT status, cancelled_at FROM tasks WHERE id = 11"
            ).fetchone()
            self.assertEqual(row[0], "cancelled")
            self.assertIsNotNone(row[1])

            # Repeated cancellation is a stable no-op.
            again = database.cancel_event("evt-drift")
            self.assertEqual(again["cancelled"], 0)
            self.assertEqual(again["counts"], result["counts"])

            # Unknown event still 404s at the API layer's None boundary.
            self.assertIsNone(database.cancel_event("does-not-exist"))
        finally:
            database.close()

    def test_drifted_database_restart_is_idempotent(self) -> None:
        _create_drifted_cancelled_column_database(self.path)
        Database(self.path).close()
        # A second start against the now-stamped file must be a pure no-op:
        # same version, same rows, and cancel still works.
        reopened = Database(self.path)
        try:
            self.assertEqual(
                reopened._conn.execute("PRAGMA user_version").fetchone()[0],
                LATEST_USER_VERSION,
            )
            result = reopened.cancel_event("evt-drift")
            self.assertEqual(result["cancelled"], 1)
            self.assertEqual(
                reopened._conn.execute(
                    "SELECT COUNT(*) FROM tasks"
                ).fetchone()[0],
                4,
            )
            Database(self.path).close()  # third start: still clean
        finally:
            reopened.close()

    def test_failed_v7_rebuild_rolls_back_without_advancing_version(self) -> None:
        _create_drifted_cancelled_column_database(self.path)

        # Make the rebuild fail at its final statement (index creation): the
        # whole step -- probe, table rebuild, data movement -- must roll back
        # atomically and user_version must stay at the previous whole step.
        original_statements = dbmod._TASKS_TABLE_REBUILD
        broken_statements = original_statements[:-1] + (
            "CREATE INDEX idx_tasks_sub_order ON no_such_table (x)",
        )
        dbmod._TASKS_TABLE_REBUILD = broken_statements
        try:
            with self.assertRaises(MigrationError):
                Database(self.path)
        finally:
            dbmod._TASKS_TABLE_REBUILD = original_statements

        conn = _raw_connect(self.path)
        try:
            # Steps 1-6 committed; step 7 left no trace.
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 6)
            self.assertIsNone(
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='tasks_new'"
                ).fetchone()
            )
            # The original table is exactly as the drifting release left it,
            # data and old (narrow) CHECK included.
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0], 4
            )
            self.assertEqual(
                conn.execute("SELECT status FROM tasks WHERE id=11").fetchone()[0],
                "pending",
            )
            self.assertEqual(
                conn.execute(
                    "SELECT cancelled_at FROM tasks WHERE id=12"
                ).fetchone()[0],
                7.75,
            )
            self.assertEqual(
                conn.execute(
                    "SELECT seq FROM sqlite_sequence WHERE name='tasks'"
                ).fetchone()[0],
                12,
            )
        finally:
            conn.close()

        # Healthy restart resumes the chain and reaches the latest version.
        database = Database(self.path)
        try:
            self.assertEqual(
                database._conn.execute("PRAGMA user_version").fetchone()[0],
                LATEST_USER_VERSION,
            )
            self.assertEqual(
                database._conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0],
                4,
            )
            result = database.cancel_event("evt-drift")
            self.assertEqual(result["cancelled"], 1)
        finally:
            database.close()

    # ------------------------------------------------------------------
    # step atomicity: failure rolls back, restart resumes
    # ------------------------------------------------------------------
    def test_failed_step_rolls_back_and_resumes_after_repair(self) -> None:
        _create_v1_database(self.path)

        def broken_v3(conn) -> None:
            # Perform a DDL change inside the step, then fail: the whole step
            # (DDL + version bump) must roll back atomically.
            conn.execute(
                "ALTER TABLE subscriptions ADD COLUMN previous_secret TEXT"
            )
            raise RuntimeError("simulated migration failure")

        broken = tuple(
            replace(m, apply=broken_v3) if m.version == 3 else m
            for m in dbmod._MIGRATIONS
        )
        original = dbmod._MIGRATIONS
        dbmod._MIGRATIONS = broken
        try:
            with self.assertRaises(MigrationError):
                Database(self.path)
        finally:
            dbmod._MIGRATIONS = original

        # Inspect the raw file: versions 1-2 committed, step 3 fully rolled
        # back (its column and user_version advance are both absent).
        conn = _raw_connect(self.path)
        try:
            self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 2)
            cols = {r[1] for r in conn.execute("PRAGMA table_info(subscriptions)")}
            self.assertIn("paused_at", cols)  # committed by step 2
            self.assertNotIn("previous_secret", cols)  # rolled back with step 3
            # Pre-existing data is untouched.
            self.assertEqual(
                conn.execute("SELECT secret FROM subscriptions").fetchone()[0],
                "s3cr3t",
            )
        finally:
            conn.close()

        # "Repair" = restart with healthy code: the chain resumes at step 3
        # and reaches the latest version.
        database = Database(self.path)
        try:
            self.assertEqual(
                database._conn.execute("PRAGMA user_version").fetchone()[0],
                LATEST_USER_VERSION,
            )
            cols = {
                r[1]
                for r in database._conn.execute(
                    "PRAGMA table_info(subscriptions)"
                )
            }
            self.assertIn("previous_secret", cols)
            self.assertIn("window_updated_at", cols)
        finally:
            database.close()

    # ------------------------------------------------------------------
    # concurrent first starts
    # ------------------------------------------------------------------
    def test_concurrent_opens_apply_each_step_exactly_once(self) -> None:
        _create_v1_database(self.path)

        applies: dict[int, int] = {}
        applies_lock = threading.Lock()

        def counted(migration: Migration) -> Migration:
            original_apply = migration.apply

            def wrapper(conn):
                result = original_apply(conn)
                with applies_lock:
                    applies[migration.version] = (
                        applies.get(migration.version, 0) + 1
                    )
                return result

            return replace(migration, apply=wrapper)

        instrumented = tuple(counted(m) for m in dbmod._MIGRATIONS)
        original = dbmod._MIGRATIONS
        dbmod._MIGRATIONS = instrumented

        databases: list[Database] = []
        errors: list[BaseException] = []
        barrier = threading.Barrier(5)

        def open_one() -> None:
            try:
                barrier.wait(timeout=10)
                databases.append(Database(self.path))
            except BaseException as exc:  # noqa: BLE001 - report any thread error
                errors.append(exc)

        threads = [threading.Thread(target=open_one) for _ in range(5)]
        try:
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)
                self.assertFalse(t.is_alive(), "migration thread hung")
        finally:
            dbmod._MIGRATIONS = original

        self.assertEqual(errors, [])
        self.assertEqual(len(databases), 5)
        try:
            # One writer per version; the losing instances re-read
            # user_version under the write lock and skip finished steps.
            self.assertEqual(applies, {v: 1 for v in range(1, LATEST_USER_VERSION + 1)})
            for d in databases:
                self.assertEqual(
                    d._conn.execute("PRAGMA user_version").fetchone()[0],
                    LATEST_USER_VERSION,
                )
            # Migrated data is readable through every connection.
            for d in databases:
                self.assertEqual(
                    d._conn.execute(
                        "SELECT COUNT(*) FROM subscriptions"
                    ).fetchone()[0],
                    1,
                )
        finally:
            for d in databases:
                d.close()

    # ------------------------------------------------------------------
    # fatal startup conditions
    # ------------------------------------------------------------------
    def test_foreign_key_violation_after_migration_refuses_start(self) -> None:
        database = Database(self.path)
        try:
            conn = database._conn
            conn.execute(
                "INSERT INTO events (id, payload, created_at) VALUES ('e1', '{}', 1)"
            )
            conn.execute(
                "INSERT INTO subscriptions (id, target_url, secret, created_at) "
                "VALUES ('s1', 'u', 'k', 1)"
            )
            conn.execute(
                """INSERT INTO tasks (event_id, subscription_id, status, attempts,
                      max_attempts, created_at, updated_at)
                   VALUES ('e1', 's1', 'pending', 0, 4, 1, 1)"""
            )
        finally:
            database.close()

        # Corrupt the file through a connection with FK enforcement off:
        # an attempts row pointing at a task id that does not exist.
        vandal = _raw_connect(self.path)
        try:
            vandal.execute(
                "INSERT INTO attempts (task_id, attempt_no, status_code, error, "
                "started_at, finished_at) VALUES (9999, 1, 500, 'x', 1, 1)"
            )
        finally:
            vandal.close()

        with self.assertRaises(MigrationError) as ctx:
            Database(self.path)
        self.assertIn("foreign key integrity check", str(ctx.exception))

        # The version is fine; only integrity gating rejected the file.
        check = _raw_connect(self.path)
        try:
            self.assertEqual(
                check.execute("PRAGMA user_version").fetchone()[0],
                LATEST_USER_VERSION,
            )
        finally:
            check.close()

    def test_newer_database_version_is_rejected(self) -> None:
        conn = _raw_connect(self.path)
        try:
            conn.execute(
                "CREATE TABLE subscriptions (id TEXT PRIMARY KEY)"
            )
            conn.execute(f"PRAGMA user_version={LATEST_USER_VERSION + 1}")
        finally:
            conn.close()
        with self.assertRaises(MigrationError):
            Database(self.path)

    # ------------------------------------------------------------------
    # API behavior survives: exercise the upgraded schema through Database
    # ------------------------------------------------------------------
    def test_api_behavior_after_upgrade(self) -> None:
        _create_v1_database(self.path)
        database = Database(self.path)
        try:
            # New features backed by the migration all function on upgraded
            # data: pause, rate policy, window, task creation and claims.
            self.assertIsNotNone(database.get_subscription("sub-aaaa0001"))
            paused = database.set_subscription_paused("sub-aaaa0001", True)
            self.assertTrue(paused["paused_at"] is not None)
            policy = database.set_rate_policy("sub-aaaa0001", 2.0, 1.0)
            self.assertEqual(policy["rate_refill_per_second"], 1.0)
            window = database.set_delivery_window(
                "sub-aaaa0001", "UTC", [1, 2, 3, 4, 5], 0, 86399
            )
            self.assertTrue(window["configured"])
            task, created = database.create_task(
                "evt-1", "sub-aaaa0001", max_attempts=4
            )
            self.assertFalse(created)  # unique (event, subscription) kept
            self.assertEqual(task["status"], "failed")  # original row returned
        finally:
            database.close()


class LifespanStartupTests(unittest.TestCase):
    """The application lifespan must not start workers when migration fails."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="wr-lifespan-")
        self.path = os.path.join(self.tmp, "bad.db")

    def tearDown(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(self.path + suffix)
            except FileNotFoundError:
                pass
        os.rmdir(self.tmp)

    def _client_with_db_path(self):
        from fastapi.testclient import TestClient

        import app.main as main

        return TestClient(main.app), main

    def test_drifted_database_cancel_endpoint_works_over_http(self) -> None:
        # End-to-end regression: a half-finished version-0 file must migrate
        # on startup, after which POST /events/{event_id}/cancel returns 200
        # and flips the pending task to 'cancelled'; a second start is a
        # no-op and the cancel call is idempotent.
        _create_drifted_cancelled_column_database(self.path)
        client, main = self._client_with_db_path()

        def _start_app():
            return patch.object(app_config, "DB_PATH", self.path), \
                patch.object(main.config, "DB_PATH", self.path), \
                patch.object(app_config, "WORKER_COUNT", 0), \
                patch.object(main.config, "WORKER_COUNT", 0)

        p1, p2, p3, p4 = _start_app()
        with p1, p2, p3, p4, client:
            self.assertEqual(
                client.app.state.db._conn.execute(
                    "PRAGMA user_version"
                ).fetchone()[0],
                LATEST_USER_VERSION,
            )
            resp = client.post("/events/evt-drift/cancel")
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertEqual(body["cancelled"], 1)
            self.assertEqual(
                body["counts"],
                {"pending": 0, "in_progress": 0, "delivered": 1,
                 "failed": 1, "cancelled": 1},
            )
            self.assertIsNotNone(body["cancelled_at"])

            status = client.get("/events/evt-drift").json()
            pending = [t for t in status["targets"] if t["status"] == "pending"]
            self.assertEqual(pending, [])
            cancelled = [t for t in status["targets"] if t["status"] == "cancelled"]
            self.assertEqual(len(cancelled), 1)
            self.assertIsNotNone(cancelled[0]["cancelled_at"])

            # Unknown event -> 404.
            self.assertEqual(client.post("/events/nope/cancel").status_code, 404)

        # Restart the app against the same file: migration is idempotent and
        # the second cancel is a stable no-op.
        client2, _main2 = self._client_with_db_path()
        q1, q2, q3, q4 = _start_app()
        with q1, q2, q3, q4, client2:
            self.assertEqual(
                client2.app.state.db._conn.execute(
                    "PRAGMA user_version"
                ).fetchone()[0],
                LATEST_USER_VERSION,
            )
            again = client2.post("/events/evt-drift/cancel")
            self.assertEqual(again.status_code, 200)
            self.assertEqual(again.json()["cancelled"], 0)
            self.assertEqual(again.json()["counts"], body["counts"])

    def test_lifespan_starts_normally_after_successful_migration(self) -> None:
        _create_v1_database(self.path)
        client, main = self._client_with_db_path()
        with patch.object(app_config, "DB_PATH", self.path), \
             patch.object(main.config, "DB_PATH", self.path):
            with client:
                resp = client.get("/healthz")
                self.assertEqual(resp.status_code, 200)
                # Workers were started against the migrated database.
                self.assertEqual(
                    client.app.state.db._conn.execute(
                        "PRAGMA user_version"
                    ).fetchone()[0],
                    LATEST_USER_VERSION,
                )

    def test_lifespan_refuses_start_on_foreign_key_violation(self) -> None:
        # A migrated file that later acquires an orphan child row must make
        # the lifespan raise before any delivery worker is created.
        database = Database(self.path)
        database.close()
        vandal = _raw_connect(self.path)
        try:
            vandal.execute(
                "INSERT INTO attempts (task_id, attempt_no, status_code, error, "
                "started_at, finished_at) VALUES (424242, 1, 500, 'x', 1, 1)"
            )
        finally:
            vandal.close()

        client, main = self._client_with_db_path()
        workers_seen = []
        original_create_task = main.asyncio.create_task

        def spy_create_task(coro, *args, **kwargs):  # pragma: no cover - must not run
            workers_seen.append(coro)
            return original_create_task(coro, *args, **kwargs)

        with patch.object(app_config, "DB_PATH", self.path), \
             patch.object(main.config, "DB_PATH", self.path), \
             patch.object(main.asyncio, "create_task", side_effect=spy_create_task):
            with self.assertRaises(MigrationError):
                with client:
                    pass  # entering the lifespan runs startup to completion
            self.assertEqual(workers_seen, [])  # no worker task was scheduled
            self.assertFalse(hasattr(client.app.state, "db"))


if __name__ == "__main__":
    unittest.main()
