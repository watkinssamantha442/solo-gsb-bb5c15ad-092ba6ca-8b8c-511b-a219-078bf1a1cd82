"""Regression tests for POST /events/{id}/cancel time semantics.

The event-level cancellation timestamp used to be stamped from the call
clock: repeating a cancel against an event with no pending tasks left
refreshed the response 'cancelled_at' to the call time, disagreeing with
the persisted task rows (and with GET /events/{id}, which derives the
event-level time from those rows). These tests pin the fixed contract:

- the cancel response and GET /events/{id} both report the most recent
  *actual* cancellation time persisted on the event's cancelled tasks;
  an event that never cancelled any task reports null (the response
  model allows it);
- a repeated cancel cancels zero tasks, keeps the original timestamp,
  and rewrites neither task rows nor circuit-breaker state;
- with mixed pending / in_progress / terminal tasks only the pending
  ones are stamped, all with the same timestamp, inside one transaction
  that stays mutually exclusive with worker claims.

Pure stdlib + the app's own FastAPI test client; run with either of

    python3 -m unittest discover -s tests -v
    python3 -m pytest tests/
"""
from __future__ import annotations

import http.server
import json
import os
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

# A private temp database must be selected before app.config (and hence
# app.main) is first imported, since config reads the environment at import.
_TMP_ROOT = tempfile.mkdtemp(prefix="wr-event-cancel-")
os.environ.setdefault("DB_PATH", os.path.join(_TMP_ROOT, "default.db"))
os.environ.setdefault("INGEST_SECRET", "test-ingest-secret")
os.environ.setdefault("WORKER_COUNT", "2")

from fastapi.testclient import TestClient  # noqa: E402

from app import config  # noqa: E402
from app.db import Database  # noqa: E402
import app.main as main  # noqa: E402

HOOK_URL = "http://127.0.0.1:1/hook"  # port 1 never accepts a connection
PAYLOAD = json.dumps({"n": 1}, separators=(",", ":"))

# Fixed clocks for the cancel calls: far apart, so a response stamped from
# the call clock (the old bug) is unmistakably different from the
# persisted cancellation time.
FIRST_CANCEL_TS = 1_700_000_000.0
SECOND_CANCEL_TS = 1_700_000_600.0


class CancelEventDatabaseTests(unittest.TestCase):
    """Database-level guarantees of Database.cancel_event (no HTTP)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=_TMP_ROOT)
        self.db = Database(os.path.join(self.tmp.name, "db.db"))

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def _sub(self) -> str:
        return self.db.create_subscription(HOOK_URL, "s")["id"]

    def _submit(self, event_id: str, sids: list[str]) -> list[int]:
        status, receipt = self.db.submit_direct_event(event_id, PAYLOAD, sids, 4)
        self.assertEqual(status, "created")
        return [r["task_id"] for r in receipt["created"]]

    def _task(self, task_id: int) -> dict:
        return self.db.get_delivery_task(task_id)["task"]

    def _breaker_rows(self) -> dict[str, dict]:
        rows = self.db._conn.execute(
            """SELECT id, breaker_state, breaker_consecutive_failures,
                      breaker_next_probe_at, breaker_probe_task_id
                 FROM subscriptions"""
        ).fetchall()
        return {r["id"]: dict(r) for r in rows}

    # ------------------------------------------------------------------
    # mixed states: only pending rows are stamped, with one timestamp
    # ------------------------------------------------------------------
    def test_mixed_states_only_pending_stamped_with_one_timestamp(self):
        sids = [self._sub() for _ in range(5)]
        t_inprog, t_delivered, t_failed, t_pend1, t_pend2 = self._submit(
            "evt-mixed", sids
        )

        now = time.time()
        # One task claimed by a worker (attempt still running) ...
        claimed = self.db.claim_due_task("w1", now)
        self.assertEqual(claimed["id"], t_inprog)
        # ... one delivered ...
        claimed = self.db.claim_due_task("w1", now)
        self.assertEqual(claimed["id"], t_delivered)
        self.db.finish_attempt(t_delivered, 1, 200, None, now, now, "delivered", None)
        # ... one terminally failed (breaker count 1, still closed) ...
        claimed = self.db.claim_due_task("w1", now)
        self.assertEqual(claimed["id"], t_failed)
        self.db.finish_attempt(t_failed, 1, 500, "boom", now, now, "failed", None)
        # ... and two still pending.

        untouched_before = {
            tid: self._task(tid) for tid in (t_inprog, t_delivered, t_failed)
        }
        breakers_before = self._breaker_rows()

        with mock.patch("time.time", return_value=FIRST_CANCEL_TS):
            result = self.db.cancel_event("evt-mixed")

        self.assertEqual(result["cancelled"], 2)
        self.assertEqual(result["cancelled_at"], FIRST_CANCEL_TS)
        self.assertEqual(
            result["counts"],
            {"pending": 0, "in_progress": 1, "delivered": 1, "failed": 1, "cancelled": 2},
        )

        # Both pending tasks carry the very same persisted timestamp; the
        # response timestamp is exactly that persisted value.
        for tid in (t_pend1, t_pend2):
            task = self._task(tid)
            self.assertEqual(task["status"], "cancelled")
            self.assertEqual(task["cancelled_at"], FIRST_CANCEL_TS)
            self.assertEqual(task["updated_at"], FIRST_CANCEL_TS)
            self.assertIsNone(task["next_attempt_at"])
            self.assertIsNone(task["claimed_by"])

        # in_progress and terminal tasks are byte-identical after the cancel.
        for tid, before in untouched_before.items():
            self.assertEqual(self._task(tid), before)
        self.assertIsNone(self._task(t_inprog)["cancelled_at"])
        self.assertEqual(self._task(t_inprog)["claimed_by"], "w1")

        # Cancelling never touches the circuit-breaker bookkeeping (the
        # failed task's count of 1 stays as finish_attempt recorded it).
        self.assertEqual(self._breaker_rows(), breakers_before)
        self.assertEqual(self._breaker_rows()[sids[2]]["breaker_consecutive_failures"], 1)

    # ------------------------------------------------------------------
    # repeated cancel: persisted time, not the call clock
    # ------------------------------------------------------------------
    def test_repeated_cancel_reports_persisted_time_not_call_time(self):
        sid = self._sub()
        (task_id,) = self._submit("evt-repeat", [sid])

        with mock.patch("time.time", return_value=FIRST_CANCEL_TS):
            first = self.db.cancel_event("evt-repeat")
        self.assertEqual(first["cancelled"], 1)
        self.assertEqual(first["cancelled_at"], FIRST_CANCEL_TS)

        with mock.patch("time.time", return_value=SECOND_CANCEL_TS):
            second = self.db.cancel_event("evt-repeat")
        self.assertEqual(second["cancelled"], 0)
        # The persisted cancellation time, not the second call's clock.
        self.assertEqual(second["cancelled_at"], FIRST_CANCEL_TS)
        self.assertEqual(second["counts"], first["counts"])

        # The repeat did not rewrite the task row at all.
        task = self._task(task_id)
        self.assertEqual(task["status"], "cancelled")
        self.assertEqual(task["cancelled_at"], FIRST_CANCEL_TS)
        self.assertEqual(task["updated_at"], FIRST_CANCEL_TS)

    # ------------------------------------------------------------------
    # never-cancelled events report NULL
    # ------------------------------------------------------------------
    def test_event_with_nothing_cancelled_reports_null(self):
        sid = self._sub()
        # All tasks terminal (delivered): nothing to cancel, none cancelled.
        (task_id,) = self._submit("evt-delivered", [sid])
        now = time.time()
        claimed = self.db.claim_due_task("w1", now)
        self.assertEqual(claimed["id"], task_id)
        self.db.finish_attempt(task_id, 1, 200, None, now, now, "delivered", None)

        result = self.db.cancel_event("evt-delivered")
        self.assertEqual(result["cancelled"], 0)
        self.assertIsNone(result["cancelled_at"])
        self.assertEqual(result["counts"]["delivered"], 1)

        # Only an in_progress task: likewise nothing ever cancelled.
        (task2,) = self._submit("evt-inprog", [sid])
        claimed = self.db.claim_due_task("w1", time.time())
        self.assertEqual(claimed["id"], task2)
        result = self.db.cancel_event("evt-inprog")
        self.assertEqual(result["cancelled"], 0)
        self.assertIsNone(result["cancelled_at"])
        self.assertEqual(result["counts"]["in_progress"], 1)

    def test_unknown_event_returns_none(self):
        self.assertIsNone(self.db.cancel_event("evt-missing"))

    # ------------------------------------------------------------------
    # half-open probe: cancel re-opens the breaker; a repeat keeps it stable
    # ------------------------------------------------------------------
    def test_cancelled_probe_reopens_breaker_and_repeat_keeps_it_stable(self):
        self.db.close()
        self.db = Database(
            os.path.join(self.tmp.name, "breaker.db"),
            breaker_failure_threshold=1,
            breaker_cooldown_seconds=30.0,
        )
        sid = self._sub()
        t0 = time.time()
        cancel_ts = t0 + 40.0
        recancel_ts = t0 + 50.0

        # Trip the breaker open with one final failure (threshold 1).
        (trip_task,) = self._submit("evt-trip", [sid])
        claimed = self.db.claim_due_task("w1", t0 + 1.0)
        self.assertEqual(claimed["id"], trip_task)
        self.db.finish_attempt(trip_task, 1, 500, "boom", t0, t0, "failed", None)
        breaker = self._breaker_rows()[sid]
        self.assertEqual(breaker["breaker_state"], "open")
        self.assertEqual(breaker["breaker_next_probe_at"], t0 + 30.0)

        # Cooldown over: the next claim becomes the single half-open probe,
        # then fails and backs off to 'pending' between probe attempts.
        (probe_task,) = self._submit("evt-probe", [sid])
        claimed = self.db.claim_due_task("w1", t0 + 31.0)
        self.assertEqual(claimed["id"], probe_task)
        self.assertEqual(self._breaker_rows()[sid]["breaker_state"], "half_open")
        self.assertEqual(self._breaker_rows()[sid]["breaker_probe_task_id"], probe_task)
        self.db.finish_attempt(probe_task, 1, 500, "boom", t0 + 31.0, t0 + 31.0,
                               "retry", t0 + 32.0)
        self.assertEqual(self._task(probe_task)["status"], "pending")

        # Cancelling the probe's event re-opens the breaker at once.
        with mock.patch("time.time", return_value=cancel_ts):
            first = self.db.cancel_event("evt-probe")
        self.assertEqual(first["cancelled"], 1)
        self.assertEqual(first["cancelled_at"], cancel_ts)
        breaker = self._breaker_rows()[sid]
        self.assertEqual(breaker["breaker_state"], "open")
        self.assertIsNone(breaker["breaker_probe_task_id"])
        self.assertEqual(breaker["breaker_next_probe_at"], cancel_ts)

        # A repeated cancel rewrites neither the task nor the breaker row.
        with mock.patch("time.time", return_value=recancel_ts):
            second = self.db.cancel_event("evt-probe")
        self.assertEqual(second["cancelled"], 0)
        self.assertEqual(second["cancelled_at"], cancel_ts)
        self.assertEqual(self._breaker_rows()[sid], breaker)
        task = self._task(probe_task)
        self.assertEqual(task["cancelled_at"], cancel_ts)
        self.assertEqual(task["updated_at"], cancel_ts)

    # ------------------------------------------------------------------
    # cancel vs worker claims: mutually exclusive, never both
    # ------------------------------------------------------------------
    def test_cancel_races_worker_claims_never_claimed_and_cancelled(self):
        sids = [self._sub() for _ in range(8)]
        self._submit("evt-race", sids)
        barrier = threading.Barrier(5)
        errors = []

        def claimer(worker_id: str):
            barrier.wait()
            try:
                # Claim until nothing is claimable; claimed tasks are left
                # 'in_progress' (the attempt is "still running").
                while self.db.claim_due_task(worker_id, time.time()) is not None:
                    pass
            except Exception as exc:  # pragma: no cover - failure reporting
                errors.append(exc)

        def canceller():
            barrier.wait()
            try:
                self.db.cancel_event("evt-race")
            except Exception as exc:  # pragma: no cover - failure reporting
                errors.append(exc)

        threads = [threading.Thread(target=claimer, args=(f"w{i}",)) for i in range(4)]
        threads.append(threading.Thread(target=canceller))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])

        rows = self.db._conn.execute(
            """SELECT id, status, claimed_by, cancelled_at FROM tasks
                WHERE event_id = ?""",
            ("evt-race",),
        ).fetchall()
        self.assertEqual(len(rows), 8)
        for row in rows:
            # Every task was either claimed by a worker or cancelled by the
            # event cancel -- never both, never left pending.
            if row["status"] == "cancelled":
                self.assertIsNotNone(row["cancelled_at"])
                self.assertIsNone(row["claimed_by"])
            else:
                self.assertEqual(row["status"], "in_progress")
                self.assertIsNotNone(row["claimed_by"])
                self.assertIsNone(row["cancelled_at"])

        # A follow-up cancel is a stable no-op with the persisted timestamp.
        again = self.db.cancel_event("evt-race")
        self.assertEqual(again["cancelled"], 0)
        self.assertEqual(
            again["counts"]["cancelled"] + again["counts"]["in_progress"], 8
        )
        cancelled_ts = {
            row["cancelled_at"] for row in rows if row["status"] == "cancelled"
        }
        if cancelled_ts:
            self.assertEqual(again["cancelled_at"], max(cancelled_ts))
        else:
            self.assertIsNone(again["cancelled_at"])


class _Receiver:
    """Minimal local webhook receiver: accepts every POST with 204."""

    def __init__(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(204)
                self.end_headers()

            def log_message(self, *args):
                pass

        self._server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/hook"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._server.shutdown()
        self._server.server_close()


class CancelEventAPITests(unittest.TestCase):
    """End-to-end contract of POST /events/{id}/cancel vs GET /events/{id}."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=_TMP_ROOT)
        config.DB_PATH = os.path.join(self.tmp.name, "api.db")
        self.client = TestClient(main.app)
        self.client.__enter__()
        self.addCleanup(self._shutdown)

    def _shutdown(self):
        self.client.__exit__(None, None, None)
        self.tmp.cleanup()

    def _create_sub(self, url: str = HOOK_URL) -> str:
        resp = self.client.post(
            "/subscriptions", json={"target_url": url, "secret": "s3cr3t"}
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()["id"]

    def _submit(self, event_id: str, sid: str, deliver_at: str | None = None):
        body = {"event_id": event_id, "subscription_ids": [sid], "payload": {"n": 1}}
        if deliver_at is not None:
            body["deliver_at"] = deliver_at
        resp = self.client.post("/events", json=body)
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp

    def test_cancel_then_get_report_same_actual_time_and_repeat_is_stable(self):
        sid = self._create_sub()
        # Scheduled in the future so no worker can claim the task first.
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        self._submit("evt-api", sid, deliver_at=future)

        first = self.client.post("/events/evt-api/cancel")
        self.assertEqual(first.status_code, 200, first.text)
        body1 = first.json()
        self.assertEqual(body1["cancelled"], 1)
        self.assertIsNotNone(body1["cancelled_at"])
        self.assertEqual(
            body1["counts"],
            {"pending": 0, "in_progress": 0, "delivered": 0, "failed": 0, "cancelled": 1},
        )

        # GET derives the very same event-level timestamp from the task rows.
        status = self.client.get("/events/evt-api").json()
        self.assertTrue(status["cancelled"])
        self.assertEqual(status["cancelled_at"], body1["cancelled_at"])
        self.assertEqual(status["targets"][0]["status"], "cancelled")
        self.assertEqual(status["targets"][0]["cancelled_at"], body1["cancelled_at"])

        # A repeated cancel is a no-op that keeps the original timestamp
        # instead of being refreshed to the call time.
        second = self.client.post("/events/evt-api/cancel")
        self.assertEqual(second.status_code, 200, second.text)
        body2 = second.json()
        self.assertEqual(body2["cancelled"], 0)
        self.assertEqual(body2["cancelled_at"], body1["cancelled_at"])
        self.assertEqual(body2["counts"], body1["counts"])

        status2 = self.client.get("/events/evt-api").json()
        self.assertEqual(status2["cancelled_at"], body1["cancelled_at"])
        self.assertEqual(status2["counts"], body1["counts"])

    def test_cancel_after_delivery_reports_null_and_get_agrees(self):
        receiver = _Receiver()
        receiver.start()
        self.addCleanup(receiver.stop)
        sid = self._create_sub(receiver.url)
        self._submit("evt-delivered", sid)

        deadline = time.time() + 10
        status = {}
        while time.time() < deadline:
            status = self.client.get("/events/evt-delivered").json()
            if status["counts"]["delivered"] == 1:
                break
            time.sleep(0.05)
        self.assertEqual(status["counts"]["delivered"], 1, status)

        # Nothing pending and nothing ever cancelled: null, not a timestamp.
        cancel = self.client.post("/events/evt-delivered/cancel")
        self.assertEqual(cancel.status_code, 200, cancel.text)
        body = cancel.json()
        self.assertEqual(body["cancelled"], 0)
        self.assertIsNone(body["cancelled_at"])
        self.assertEqual(body["counts"]["delivered"], 1)

        status = self.client.get("/events/evt-delivered").json()
        self.assertFalse(status["cancelled"])
        self.assertIsNone(status["cancelled_at"])

    def test_cancel_unknown_event_is_404(self):
        resp = self.client.post("/events/evt-missing/cancel")
        self.assertEqual(resp.status_code, 404)

    def test_cancel_response_model_allows_null_cancelled_at(self):
        # The response model must expose cancelled_at as nullable so an
        # event that never cancelled a task can report null.
        schema = self.client.get("/openapi.json").json()
        prop = schema["components"]["schemas"]["EventCancelResult"]["properties"][
            "cancelled_at"
        ]
        types = {item.get("type") for item in prop.get("anyOf", [])}
        self.assertIn("string", types)
        self.assertIn("null", types)


if __name__ == "__main__":
    unittest.main()
