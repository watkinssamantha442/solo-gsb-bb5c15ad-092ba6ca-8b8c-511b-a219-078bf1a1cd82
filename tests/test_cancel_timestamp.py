"""Regression tests for POST /events/{id}/cancel cancellation-time semantics.

The cancel endpoint used to answer ``cancelled_at`` with *the wall-clock
instant of every call* (the ``now`` of the transaction), regardless of
whether that call actually cancelled any task. Repeating the call on an
event that no longer had any ``pending`` task therefore rewrote the
response's ``cancelled_at`` to the repeat instant, even though the
persisted ``cancelled_at`` of the already-cancelled tasks was older and
never changed. The API response and GET /events/{id} disagreed about when
the event was last cancelled.

These tests pin the repaired contract at both the persistence and HTTP
boundaries:

- the event-level ``cancelled_at`` is derived from the cancelled tasks'
  persisted timestamps (MAX), identical to what GET /events/{id} reports;
  all tasks cancelled by one call share one identical timestamp;
- a repeat call that cancels nothing returns the very same ``cancelled_at``
  (it is not refreshed to the call instant), ``cancelled`` is 0, and
  neither task rows nor breaker state are rewritten;
- an event whose tasks were all in progress/terminal at every cancel call
  gets ``cancelled_at`` null (and the response model serializes null);
- with a mix of pending, in_progress and terminal tasks only the pending
  rows move to 'cancelled' (same timestamp, updated_at changed for them
  alone); in_progress / delivered / failed rows keep status and
  updated_at untouched;
- canceling the unique pending probe task of a half-open breaker repairs
  the breaker to open, but a repeat call never moves it again;
- cancellation and worker claiming are mutually exclusive: under
  concurrent cancel/claim contention every task ends claimed or cancelled
  exactly once, never both;
- the half-open-probe repair path and the unknown-event 404 contract are
  preserved.

Pure stdlib + the app's own FastAPI test client; run with either of

    python3 -m unittest discover -s tests -v
    python3 -m pytest tests/
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

# A private temp database must be selected before app.config (and hence
# app.main) is first imported, since config reads the environment at import.
_TMP_ROOT = tempfile.mkdtemp(prefix="wr-cancel-time-")
os.environ.setdefault("DB_PATH", os.path.join(_TMP_ROOT, "default.db"))
os.environ.setdefault("INGEST_SECRET", "test-ingest-secret")
# The API test cases pin WORKER_COUNT to 0 so no background worker races the
# deliberately constructed task states; the claim/cancel mutual-exclusion
# contract is exercised explicitly in the Database test class.
os.environ.setdefault("WORKER_COUNT", "0")

from fastapi.testclient import TestClient  # noqa: E402

from app import config  # noqa: E402
from app.db import Database  # noqa: E402
import app.main as main  # noqa: E402

HOOK_URL = "http://127.0.0.1:1/hook"


def _task_rows(db: Database, event_id: str) -> list[dict]:
    rows = db._conn.execute(
        """SELECT id, subscription_id, status, attempts, cancelled_at, updated_at
             FROM tasks WHERE event_id = ? ORDER BY id""",
        (event_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _set_task_state(db: Database, task_id: int, status: str, **cols) -> None:
    """Force a task into a state (test fixture only).

    Mirrors what the worker would persist: a claimed task is in_progress
    with a claimed_by, a terminal task has next_attempt_at NULL and a
    result marker. Extra columns (cancelled_at/updated_at) may be supplied.
    """
    assignments = {"status": status}
    if status == "in_progress":
        assignments.update(claimed_by="fixture-worker")
    else:
        assignments.update(claimed_by=None)
    if status in ("delivered", "failed"):
        assignments.update(next_attempt_at=None, result=status)
    assignments.update(cols)
    keys = ", ".join(f"{k} = ?" for k in assignments)
    db._conn.execute(f"UPDATE tasks SET {keys} WHERE id = ?", (*assignments.values(), task_id))


# Epoch stamps deliberately in the past so they can never equal time.time()
# during a test; distinct and far apart so float equality is unambiguous.
PAST_T1 = 1_000_000_000.0   # 2001-09-09T01:46:40Z
PAST_T2 = 1_000_001_000.0   # ~17 minutes later


class CancelEventTimestampDatabaseTests(unittest.TestCase):
    """Persistence-layer guarantees: no HTTP, no workers, raw state checks."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=_TMP_ROOT)
        self.path = os.path.join(self.tmp.name, "db.db")
        self.db = Database(self.path)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def _event_with_tasks(self, event_id: str, n: int = 1) -> list[str]:
        sids = [self.db.create_subscription(HOOK_URL, "s")["id"] for _ in range(n)]
        self.db.create_event(event_id, "{}")
        for sid in sids:
            self.db.create_task(event_id, sid, max_attempts=4)
        return sids

    # ------------------------------------------------------------------
    # cancelled_at is derived from persisted rows, not stamped on read
    # ------------------------------------------------------------------
    def test_cancelled_at_comes_from_persisted_tasks_and_is_stable_on_repeat(self):
        self._event_with_tasks("evt-db-stable", 2)

        first = self.db.cancel_event("evt-db-stable")
        self.assertEqual(first["cancelled"], 2)
        self.assertIsNotNone(first["cancelled_at"])
        stamped = {
            r["cancelled_at"]
            for r in _task_rows(self.db, "evt-db-stable")
        }
        self.assertEqual(stamped, {first["cancelled_at"]})

        # Wait across the float-clock resolution so the repeat transaction's
        # own time.time() is provably different; it must still not leak into
        # the response.
        time.sleep(0.01)
        second = self.db.cancel_event("evt-db-stable")
        self.assertEqual(second["cancelled"], 0)
        self.assertEqual(second["counts"], first["counts"])
        self.assertEqual(second["cancelled_at"], first["cancelled_at"])
        # And the rows keep their original timestamps.
        self.assertEqual(
            {r["cancelled_at"] for r in _task_rows(self.db, "evt-db-stable")},
            {first["cancelled_at"]},
        )

    def test_never_cancelled_event_reports_null_even_after_calls(self):
        # One event with only a finished task: a cancel call succeeds (the
        # event exists) but cancels nothing, so cancelled_at is NULL.
        self._event_with_tasks("evt-db-null", 1)
        task_id = _task_rows(self.db, "evt-db-null")[0]["id"]
        with self.db._immediate():
            _set_task_state(self.db, task_id, "delivered", updated_at=PAST_T1)

        result = self.db.cancel_event("evt-db-null")
        self.assertEqual(result["cancelled"], 0)
        self.assertIsNone(result["cancelled_at"])
        self.assertEqual(result["counts"]["delivered"], 1)
        self.assertEqual(result["counts"]["cancelled"], 0)

        # A repeat call stays null and the finished row is untouched.
        again = self.db.cancel_event("evt-db-null")
        self.assertEqual(again["cancelled"], 0)
        self.assertIsNone(again["cancelled_at"])
        row = _task_rows(self.db, "evt-db-null")[0]
        self.assertEqual(row["status"], "delivered")
        self.assertEqual(row["updated_at"], PAST_T1)

    # ------------------------------------------------------------------
    # mixed states: only pending rows move, and they share one stamp
    # ------------------------------------------------------------------
    def test_only_pending_tasks_are_cancelled_with_one_shared_timestamp(self):
        event_id = "evt-db-mixed"
        sids = self._event_with_tasks(event_id, 5)
        rows = {r["subscription_id"]: r for r in _task_rows(self.db, event_id)}

        # Arrange: one in_progress (claimed), one delivered, one failed,
        # two still pending -- one of them a backoff/retry pending task
        # (attempts > 0).
        with self.db._immediate():
            _set_task_state(self.db, rows[sids[0]]["id"], "in_progress",
                            updated_at=PAST_T1)
            _set_task_state(self.db, rows[sids[1]]["id"], "delivered",
                            attempts=1, updated_at=PAST_T1)
            _set_task_state(self.db, rows[sids[2]]["id"], "failed",
                            attempts=4, updated_at=PAST_T2)
            self.db._conn.execute(
                "UPDATE tasks SET attempts = 2, next_attempt_at = ?, updated_at = ? WHERE id = ?",
                (PAST_T2, PAST_T2, rows[sids[3]]["id"]),
            )
        # sids[4] is a fresh pending task.

        result = self.db.cancel_event(event_id)
        self.assertEqual(result["cancelled"], 2)
        self.assertEqual(
            result["counts"],
            {"pending": 0, "in_progress": 1, "delivered": 1, "failed": 1,
             "cancelled": 2},
        )
        self.assertIsNotNone(result["cancelled_at"])

        after = {r["subscription_id"]: r for r in _task_rows(self.db, event_id)}
        # The two pending tasks are cancelled together with one timestamp and
        # one updated_at (the cancel instant); the backoff task loses its
        # retry schedule and claim marker.
        cancelled_rows = [after[sids[3]], after[sids[4]]]
        self.assertEqual({r["status"] for r in cancelled_rows}, {"cancelled"})
        self.assertEqual({r["cancelled_at"] for r in cancelled_rows},
                         {result["cancelled_at"]})
        self.assertEqual({r["updated_at"] for r in cancelled_rows},
                         {result["cancelled_at"]})
        raw = self.db._conn.execute(
            "SELECT next_attempt_at, claimed_by FROM tasks WHERE id IN (?, ?)",
            (after[sids[3]]["id"], after[sids[4]]["id"]),
        ).fetchall()
        self.assertEqual({(r["next_attempt_at"], r["claimed_by"]) for r in raw},
                         {(None, None)})

        # The non-pending rows survive untouched: statuses and updated_at
        # stay exactly as arranged.
        self.assertEqual(after[sids[0]]["status"], "in_progress")
        self.assertEqual(after[sids[0]]["updated_at"], PAST_T1)
        self.assertEqual(after[sids[1]]["status"], "delivered")
        self.assertEqual(after[sids[1]]["updated_at"], PAST_T1)
        self.assertIsNone(after[sids[1]]["cancelled_at"])
        self.assertEqual(after[sids[2]]["status"], "failed")
        self.assertEqual(after[sids[2]]["updated_at"], PAST_T2)
        self.assertIsNone(after[sids[2]]["cancelled_at"])

        # Repeat: zero newly cancelled, stable derived timestamp and counts,
        # and no row's updated_at/cancelled_at moves.
        before = {(r["id"], r["status"], r["cancelled_at"], r["updated_at"])
                  for r in _task_rows(self.db, event_id)}
        time.sleep(0.01)
        again = self.db.cancel_event(event_id)
        self.assertEqual(again["cancelled"], 0)
        self.assertEqual(again["cancelled_at"], result["cancelled_at"])
        self.assertEqual(again["counts"], result["counts"])
        after_again = {(r["id"], r["status"], r["cancelled_at"], r["updated_at"])
                       for r in _task_rows(self.db, event_id)}
        self.assertEqual(after_again, before)

    def test_later_cancel_batch_sets_latest_time_but_keeps_old_stamps(self):
        # Two pending tasks cancelled at t_a; one task claimed (in_progress)
        # before that call survives; when it later becomes pending again
        # (crash-recovery style reset) a second cancel stamps it at t_b.
        # The event-level cancelled_at becomes t_b while the first two
        # tasks keep t_a.
        sids = self._event_with_tasks("evt-db-batches", 3)
        rows = {r["subscription_id"]: r for r in _task_rows(self.db, "evt-db-batches")}
        t_a = PAST_T1
        # Claim the third task *before* the first cancel, so it is
        # in_progress at t_a and must be left alone.
        with self.db._immediate():
            _set_task_state(self.db, rows[sids[2]]["id"], "in_progress",
                            updated_at=t_a)
        with _patched_time(t_a):
            first = self.db.cancel_event("evt-db-batches")
        self.assertEqual(first["cancelled"], 2)
        self.assertEqual(first["cancelled_at"], t_a)
        self.assertEqual(first["counts"]["in_progress"], 1)

        # Simulate a restart: the interrupted in-flight task is requeued to
        # pending; a later cancel then catches it and stamps it at t_b.
        self.db.requeue_in_progress()
        t_b = PAST_T2
        with _patched_time(t_b):
            second = self.db.cancel_event("evt-db-batches")
        self.assertEqual(second["cancelled"], 1)
        self.assertEqual(second["cancelled_at"], t_b)

        after = {r["subscription_id"]: r for r in _task_rows(self.db, "evt-db-batches")}
        self.assertEqual(after[sids[0]]["cancelled_at"], t_a)
        self.assertEqual(after[sids[1]]["cancelled_at"], t_a)
        self.assertEqual(after[sids[2]]["cancelled_at"], t_b)

    # ------------------------------------------------------------------
    # half-open breaker repair happens once, never on a repeat
    # ------------------------------------------------------------------
    def test_half_open_probe_cancel_repairs_breaker_and_repeat_leaves_it(self):
        # Two subscriptions in one event. Trip the first breaker, then put it
        # into half_open whose unique probe is this event's pending task.
        probe_sid = self.db.create_subscription(HOOK_URL, "s")["id"]
        other_sid = self.db.create_subscription(HOOK_URL, "s")["id"]
        event_id = "evt-db-probe"
        self.db.create_event(event_id, "{}")
        probe_task, created = self.db.create_task(event_id, probe_sid, max_attempts=4)
        self.assertTrue(created)
        _, created = self.db.create_task(event_id, other_sid, max_attempts=4)
        self.assertTrue(created)
        with self.db._immediate():
            self.db._conn.execute(
                """UPDATE subscriptions
                      SET breaker_state = 'half_open', breaker_consecutive_failures = 3,
                          breaker_next_probe_at = NULL, breaker_probe_task_id = ?
                    WHERE id = ?""",
                (probe_task["id"], probe_sid),
            )

        result = self.db.cancel_event(event_id)
        self.assertEqual(result["cancelled"], 2)
        probe_row = self.db._conn.execute(
            "SELECT breaker_state, breaker_next_probe_at, breaker_probe_task_id "
            "FROM subscriptions WHERE id = ?",
            (probe_sid,),
        ).fetchone()
        self.assertEqual(probe_row["breaker_state"], "open")
        self.assertIsNone(probe_row["breaker_probe_task_id"])
        self.assertEqual(probe_row["breaker_next_probe_at"], result["cancelled_at"])

        # The repair is one-shot: a repeat no-op must not rewrite the
        # breaker's probe time even though wall time has advanced.
        time.sleep(0.01)
        again = self.db.cancel_event(event_id)
        self.assertEqual(again["cancelled"], 0)
        probe_row_2 = self.db._conn.execute(
            "SELECT breaker_state, breaker_next_probe_at, breaker_probe_task_id "
            "FROM subscriptions WHERE id = ?",
            (probe_sid,),
        ).fetchone()
        self.assertEqual(probe_row_2["breaker_state"], "open")
        self.assertEqual(
            probe_row_2["breaker_next_probe_at"], probe_row["breaker_next_probe_at"]
        )
        self.assertEqual(again["cancelled_at"], result["cancelled_at"])

    # ------------------------------------------------------------------
    # unknown event
    # ------------------------------------------------------------------
    def test_unknown_event_returns_none(self):
        self.assertIsNone(self.db.cancel_event("no-such-event"))

    # ------------------------------------------------------------------
    # mutual exclusion with worker claims
    # ------------------------------------------------------------------
    def test_claim_and_cancel_are_mutually_exclusive_under_the_write_lock(self):
        # Two contenders -- one worker claim and one cancel -- are queued on
        # the connection lock while a third thread holds it. Whichever wins
        # after release, the loser must observe the winner's committed state
        # rather than an interleaved one: a claimed task is never cancelled,
        # and a cancelled task is never claimed. Repeated across many rounds
        # with fresh events so both winner orderings occur.
        for round_no in range(30):
            event_id = f"evt-db-mutex-{round_no}"
            self._event_with_tasks(event_id, 1)
            outcomes: dict[str, object] = {}
            out_lock = threading.Lock()
            held = threading.Event()
            start = threading.Barrier(2)

            def hold_lock():
                with self.db._lock:
                    held.set()
                    time.sleep(0.02)

            def claim():
                start.wait()
                held.wait()
                task = self.db.claim_due_task("mutex-worker", time.time())
                with out_lock:
                    outcomes["claim"] = task

            def cancel():
                start.wait()
                held.wait()
                outcomes["cancel"] = self.db.cancel_event(event_id)

            holder = threading.Thread(target=hold_lock)
            holder.start()
            t_claim = threading.Thread(target=claim)
            t_cancel = threading.Thread(target=cancel)
            t_claim.start()
            t_cancel.start()
            holder.join()
            t_claim.join()
            t_cancel.join()

            claimed_task = outcomes["claim"]
            cancel_result = outcomes["cancel"]
            self.assertIsNotNone(cancel_result)
            rows = _task_rows(self.db, event_id)
            self.assertEqual(len(rows), 1)
            self.assertIn(rows[0]["status"], ("in_progress", "cancelled"))
            if claimed_task is not None:
                # Claim won: the cancel then finds nothing pending.
                self.assertEqual(rows[0]["status"], "in_progress")
                self.assertEqual(cancel_result["cancelled"], 0)
                self.assertIsNone(cancel_result["cancelled_at"])
            else:
                # Cancel won: the claim then skips the cancelled row.
                self.assertEqual(rows[0]["status"], "cancelled")
                self.assertEqual(cancel_result["cancelled"], 1)
                self.assertEqual(rows[0]["cancelled_at"],
                                 cancel_result["cancelled_at"])

    def test_concurrent_cancel_and_claim_never_double_handle_a_task(self):
        self._event_with_tasks("evt-db-race", 12)
        # Deterministic mixed outcome with genuine lock contention: four
        # workers claim the first four tasks (committed in_progress). The
        # cancel threads are then queued on the connection lock behind a
        # short-lived lock holder while the eight remaining claim attempts
        # are released at the same instant, so the eight still-pending tasks
        # are divided between claims and the cancel sweep under real
        # contention. Every task must end claimed exactly once or cancelled
        # exactly once -- never both.
        claimed = []
        for _ in range(4):
            task = self.db.claim_due_task("race-worker", time.time())
            self.assertIsNotNone(task)
            claimed.append(task["id"])
        self.assertEqual(len(set(claimed)), 4)

        cancelled_results = []
        list_lock = threading.Lock()
        cancel_queued = threading.Event()
        release = threading.Event()
        queued = {"n": 0}
        queue_lock = threading.Lock()

        def hold_then_release():
            # Hold the connection lock just long enough for all four cancel
            # threads to block on it, then release so the queued cancels and
            # the simultaneously-released claim threads contend for real.
            with self.db._lock:
                cancel_queued.wait()
                time.sleep(0.05)
            release.set()

        def claim(_):
            release.wait()
            task = self.db.claim_due_task("race-worker", time.time())
            if task is not None:
                with list_lock:
                    claimed.append(task["id"])

        def cancel(_):
            with queue_lock:
                queued["n"] += 1
                if queued["n"] == 4:
                    cancel_queued.set()
            release.wait()
            result = self.db.cancel_event("evt-db-race")
            with list_lock:
                cancelled_results.append(result)

        holder = threading.Thread(target=hold_then_release)
        holder.start()
        cancel_threads = [threading.Thread(target=cancel, args=(i,))
                          for i in range(4)]
        for t in cancel_threads:
            t.start()
        # Wait until the cancels are blocked behind the holder, then fire the
        # eight claim attempts at the same release point.
        cancel_queued.wait()
        with ThreadPoolExecutor(max_workers=8) as pool:
            claim_futs = [pool.submit(claim, i) for i in range(8)]
            release.wait()
            for f in claim_futs:
                f.result()
        for t in cancel_threads:
            t.join()
        holder.join()

        rows = _task_rows(self.db, "evt-db-race")
        statuses = [r["status"] for r in rows]
        self.assertEqual(len(rows), 12)
        # Every task reached exactly one of the two outcomes; none is lost,
        # none is pending forever, none is both (claimed rows never later
        # flipped to cancelled).
        self.assertTrue(all(s in ("in_progress", "cancelled") for s in statuses),
                        statuses)
        self.assertEqual(
            sorted(r["id"] for r in rows if r["status"] == "in_progress"),
            sorted(set(claimed)),
        )
        cancelled_id_set = {
            r["id"] for r in rows if r["status"] == "cancelled"
        }
        self.assertTrue(cancelled_id_set.isdisjoint(set(claimed)))
        self.assertEqual(
            sum(c["cancelled"] for c in cancelled_results), len(cancelled_id_set)
        )
        # Eight tasks were pending when contention started, so at least one
        # transaction must have observed and swept them.
        self.assertTrue(cancelled_id_set)
        # Every non-null cancelled_at the cancel responses report is a stamp
        # actually persisted on some cancelled row (the response is derived,
        # never a fresh wall-clock value); and NULL can only accompany a call
        # that cancelled nothing. The final derived timestamp is checked
        # against MAX(persisted) below.
        persisted_stamps = {r["cancelled_at"] for r in rows if r["cancelled_at"] is not None}
        for c in cancelled_results:
            if c["cancelled_at"] is not None:
                self.assertIn(c["cancelled_at"], persisted_stamps)
            else:
                self.assertEqual(c["cancelled"], 0)
            self.assertEqual(c["counts"]["cancelled"]
                             + c["counts"]["in_progress"]
                             + c["counts"]["pending"]
                             + c["counts"]["delivered"]
                             + c["counts"]["failed"], 12)

        # A final cancel after the dust settles cancels nothing and reports
        # the latest persisted stamp (not a fresh clock time).
        time.sleep(0.01)
        final = self.db.cancel_event("evt-db-race")
        self.assertEqual(final["cancelled"], 0)
        if cancelled_id_set:
            self.assertEqual(final["cancelled_at"], max(persisted_stamps))
        else:
            self.assertIsNone(final["cancelled_at"])


class _patched_time:
    """Context manager making time.time() return a fixed epoch within db.py."""

    def __init__(self, fixed: float):
        self._fixed = fixed
        self._real = None

    def __enter__(self):
        import app.db as dbmod

        self._real = dbmod.time.time
        dbmod.time.time = lambda: self._fixed
        return self

    def __exit__(self, *exc):
        import app.db as dbmod

        dbmod.time.time = self._real
        return False


class CancelEventTimestampAPITests(unittest.TestCase):
    """End-to-end HTTP contract through the real FastAPI lifespan.

    WORKER_COUNT is pinned to 0 so the deliberately constructed task states
    (in_progress/terminal rows) cannot be moved by background workers.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=_TMP_ROOT)
        self._saved_workers = config.WORKER_COUNT
        config.WORKER_COUNT = 0
        config.DB_PATH = os.path.join(self.tmp.name, "api.db")
        self.client = TestClient(main.app)
        self.client.__enter__()
        self.db: Database = self.client.app.state.db
        self.addCleanup(self._shutdown)

    def _shutdown(self):
        self.client.__exit__(None, None, None)
        config.WORKER_COUNT = self._saved_workers
        self.tmp.cleanup()

    def _create_sub(self) -> str:
        resp = self.client.post(
            "/subscriptions", json={"target_url": HOOK_URL, "secret": "s3cr3t"}
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()["id"]

    def _submit_event(self, event_id: str, sids: list[str]):
        resp = self.client.post(
            "/events",
            json={"event_id": event_id, "subscription_ids": sids,
                  "payload": {"n": 1}},
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()

    def _force_states(self, event_id: str, mapping: dict[str, tuple]) -> None:
        """mapping: subscription_id -> (status, kwargs for _set_task_state)."""
        by_sid = {r["subscription_id"]: r for r in _task_rows(self.db, event_id)}
        with self.db._immediate():
            for sid, (status, kwargs) in mapping.items():
                _set_task_state(self.db, by_sid[sid]["id"], status, **kwargs)

    # ------------------------------------------------------------------
    # mixed pending/in_progress/terminal: only pending rows cancelled
    # ------------------------------------------------------------------
    def test_cancel_mixed_states_only_pending_and_matches_get(self):
        sids = [self._create_sub() for _ in range(5)]
        event_id = "evt-api-mixed"
        self._submit_event(event_id, sids)

        self._force_states(event_id, {
            sids[0]: ("in_progress", {"updated_at": PAST_T1}),
            sids[1]: ("delivered", {"attempts": 1, "updated_at": PAST_T1}),
            sids[2]: ("failed", {"attempts": 4, "updated_at": PAST_T2}),
        })
        # sids[3], sids[4] stay pending.
        non_pending_before = {
            r["subscription_id"]: (r["status"], r["updated_at"])
            for r in _task_rows(self.db, event_id)
            if r["status"] != "pending"
        }

        resp = self.client.post(f"/events/{event_id}/cancel")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["event_id"], event_id)
        self.assertEqual(body["cancelled"], 2)
        self.assertEqual(
            body["counts"],
            {"pending": 0, "in_progress": 1, "delivered": 1, "failed": 1,
             "cancelled": 2},
        )
        self.assertIsNotNone(body["cancelled_at"])

        # GET /events/{id} computes the same event-level time.
        status = self.client.get(f"/events/{event_id}")
        self.assertEqual(status.status_code, 200)
        detail = status.json()
        self.assertTrue(detail["cancelled"])
        self.assertEqual(detail["cancelled_at"], body["cancelled_at"])
        self.assertEqual(detail["counts"], body["counts"])

        targets = {t["subscription_id"]: t for t in detail["targets"]}
        cancelled_targets = [targets[sids[3]], targets[sids[4]]]
        self.assertEqual({t["status"] for t in cancelled_targets}, {"cancelled"})
        self.assertEqual(
            {t["cancelled_at"] for t in cancelled_targets},
            {body["cancelled_at"]},
        )
        for sid, state in ((sids[0], "in_progress"), (sids[1], "delivered"),
                          (sids[2], "failed")):
            self.assertEqual(targets[sid]["status"], state)
            self.assertIsNone(targets[sid]["cancelled_at"])
        # Non-pending rows were never rewritten.
        for sid, (st, updated) in non_pending_before.items():
            row = next(r for r in _task_rows(self.db, event_id)
                       if r["subscription_id"] == sid)
            self.assertEqual(row["status"], st)
            self.assertEqual(row["updated_at"], updated)
            self.assertIsNone(row["cancelled_at"])

        # Repeat call: same body, and GET stays identical too.
        time.sleep(0.01)
        repeat = self.client.post(f"/events/{event_id}/cancel")
        self.assertEqual(repeat.status_code, 200)
        repeat_body = repeat.json()
        self.assertEqual(repeat_body["cancelled"], 0)
        self.assertEqual(repeat_body["cancelled_at"], body["cancelled_at"])
        self.assertEqual(repeat_body["counts"], body["counts"])
        detail_again = self.client.get(f"/events/{event_id}").json()
        self.assertEqual(detail_again["cancelled_at"], body["cancelled_at"])

    # ------------------------------------------------------------------
    # repeat on an event with no pending tasks must not refresh the time
    # ------------------------------------------------------------------
    def test_repeat_cancel_keeps_persisted_time_not_call_time(self):
        sid = self._create_sub()
        event_id = "evt-api-repeat"
        self._submit_event(event_id, [sid])

        first = self.client.post(f"/events/{event_id}/cancel")
        self.assertEqual(first.status_code, 200)
        first_body = first.json()
        self.assertEqual(first_body["cancelled"], 1)
        first_time = first_body["cancelled_at"]
        self.assertIsNotNone(first_time)

        time.sleep(0.02)  # guarantee the second call has a later wall clock
        second = self.client.post(f"/events/{event_id}/cancel")
        self.assertEqual(second.status_code, 200)
        second_body = second.json()
        self.assertEqual(second_body["cancelled"], 0)
        self.assertEqual(second_body["cancelled_at"], first_time)

        time.sleep(0.02)
        third = self.client.post(f"/events/{event_id}/cancel")
        third_body = third.json()
        self.assertEqual(third_body["cancelled"], 0)
        self.assertEqual(third_body["cancelled_at"], first_time)

        detail = self.client.get(f"/events/{event_id}").json()
        self.assertEqual(detail["cancelled_at"], first_time)

        # The single task row keeps the single original stamp.
        rows = _task_rows(self.db, event_id)
        self.assertEqual(len(rows), 1)
        persisted_iso = datetime.fromtimestamp(
            rows[0]["cancelled_at"], tz=timezone.utc
        ).isoformat()
        self.assertEqual(persisted_iso, first_time)

    # ------------------------------------------------------------------
    # nothing ever cancelled -> cancelled_at null (model is nullable)
    # ------------------------------------------------------------------
    def test_cancel_with_only_in_progress_and_terminal_tasks_is_null(self):
        sids = [self._create_sub(), self._create_sub()]
        event_id = "evt-api-null"
        self._submit_event(event_id, sids)
        self._force_states(event_id, {
            sids[0]: ("in_progress", {"updated_at": PAST_T1}),
            sids[1]: ("delivered", {"attempts": 1, "updated_at": PAST_T1}),
        })

        resp = self.client.post(f"/events/{event_id}/cancel")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["cancelled"], 0)
        self.assertIsNone(body["cancelled_at"])
        self.assertEqual(
            body["counts"],
            {"pending": 0, "in_progress": 1, "delivered": 1, "failed": 0,
             "cancelled": 0},
        )

        # GET agrees: cancelled=false (no cancelled task) and a null time.
        detail = self.client.get(f"/events/{event_id}").json()
        self.assertFalse(detail["cancelled"])
        self.assertIsNone(detail["cancelled_at"])

        # Still null on repeat, rows untouched.
        repeat = self.client.post(f"/events/{event_id}/cancel").json()
        self.assertEqual(repeat["cancelled"], 0)
        self.assertIsNone(repeat["cancelled_at"])
        rows = _task_rows(self.db, event_id)
        self.assertEqual({r["status"] for r in rows}, {"in_progress", "delivered"})
        self.assertTrue(all(r["cancelled_at"] is None for r in rows))

    # ------------------------------------------------------------------
    # the original bug: a repeat after all pending work drained used to
    # answer the repeat wall-clock instant
    # ------------------------------------------------------------------
    def test_repeat_after_tasks_drained_does_not_refresh_time(self):
        sids = [self._create_sub(), self._create_sub()]
        event_id = "evt-api-drain"
        self._submit_event(event_id, sids)

        first = self.client.post(f"/events/{event_id}/cancel").json()
        self.assertEqual(first["cancelled"], 2)
        stamp = first["cancelled_at"]

        # All tasks are terminal ('cancelled') already; a further call is the
        # exact scenario that used to overwrite cancelled_at.
        time.sleep(0.02)
        again = self.client.post(f"/events/{event_id}/cancel").json()
        self.assertEqual(again["cancelled"], 0)
        self.assertEqual(again["cancelled_at"], stamp)
        self.assertEqual(again["counts"]["cancelled"], 2)

    def test_unknown_event_cancel_is_404(self):
        resp = self.client.post("/events/no-such-event/cancel")
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
