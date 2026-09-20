"""Concurrent regression tests for POST /events idempotency/conflict handling.

The submission of a direct event used to span several independent
transactions (subscription existence, cross-channel/topic check, payload
idempotency check, event insert, per-subscription task inserts). Two
concurrent submissions of the same event_id could therefore interleave:
the second event insert hit SQLite's PRIMARY KEY violation and surfaced as
HTTP 500, and a direct submission racing a topic publication could enqueue
its tasks against the topic's recipient set.

These tests pin the atomic-transaction contract:

- identical concurrent replays create exactly one event and one task per
  distinct (event, subscription), exactly one request answers 201, every
  other request answers 200 with the very same task ids, and the first
  request's deliver_at is never rewritten by a later replay;
- the same event_id with a different payload has one winner (201) and every
  loser stably answers 409 -- no 500, and a losing transaction leaves no
  event and no partial tasks;
- a direct submission racing a topic publication for one id resolves
  deterministically: exactly one channel wins, the other answers 409, and
  the resulting task set belongs to the winner alone (no cross-channel
  recipients), at every level from the Database boundary to HTTP;
- the sequential status contract (400 unknown subscriptions, 200 exact
  replay, 409 payload conflict, 409 cross-channel conflict, nothing
  persisted on failure) is preserved.

Pure stdlib + the app's own FastAPI test client; run with either of

    python3 -m unittest discover -s tests -v
    python3 -m pytest tests/
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

# A private temp database must be selected before app.config (and hence
# app.main) is first imported, since config reads the environment at import.
_TMP_ROOT = tempfile.mkdtemp(prefix="wr-events-race-")
os.environ.setdefault("DB_PATH", os.path.join(_TMP_ROOT, "default.db"))
os.environ.setdefault("INGEST_SECRET", "test-ingest-secret")
# Delivery targets never accept a connection (port 1); with deliver_at set in
# the future the background workers cannot claim the tasks anyway.
os.environ.setdefault("WORKER_COUNT", "2")

from fastapi.testclient import TestClient  # noqa: E402

from app import config  # noqa: E402
from app.db import Database  # noqa: E402
import app.main as main  # noqa: E402

HOOK_URL = "http://127.0.0.1:1/hook"
FUTURE = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
FUTURE2 = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()


def _task_map(body: dict) -> dict[str, dict]:
    refs = {}
    for section in ("created", "duplicates"):
        for ref in body[section]:
            refs[ref["subscription_id"]] = ref
    return refs


def _all_task_refs(body: dict) -> list[dict]:
    return body["created"] + body["duplicates"]


class SubmitEventConcurrencyAPITests(unittest.TestCase):
    """End-to-end HTTP races through the real FastAPI lifespan."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=_TMP_ROOT)
        config.DB_PATH = os.path.join(self.tmp.name, "api.db")
        # One shared TestClient is exercised from many threads; the app's
        # submission path is a synchronous def endpoint, so requests run in
        # Starlette's threadpool and truly overlap.
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

    def _topic_members(self, topic: str, sids: list[str]) -> None:
        for sid in sids:
            resp = self.client.put(
                f"/subscriptions/{sid}/topics", json={"topics": [topic]}
            )
            self.assertEqual(resp.status_code, 200, resp.text)

    def _submit(self, event_id: str, payload: dict, sids: list[str], deliver_at=None):
        body = {"event_id": event_id, "subscription_ids": sids, "payload": payload}
        if deliver_at is not None:
            body["deliver_at"] = deliver_at
        return self.client.post("/events", json=body)

    # ------------------------------------------------------------------
    # identical concurrent replays
    # ------------------------------------------------------------------
    def test_identical_concurrent_replays_create_one_task_set(self):
        sids = [self._create_sub() for _ in range(3)]
        event_id = "evt-identical"
        payload = {"n": 1}
        barrier = threading.Barrier(len(sids) * 3)

        def fire(_):
            barrier.wait()
            return self._submit(event_id, payload, sids, deliver_at=FUTURE)

        with ThreadPoolExecutor(max_workers=9) as pool:
            responses = list(pool.map(fire, range(9)))

        statuses = sorted(r.status_code for r in responses)
        self.assertNotIn(500, statuses, [r.text for r in responses])
        # Exactly one first-time creation; every other concurrent request is
        # an idempotent replay.
        self.assertEqual(statuses.count(201), 1, statuses)
        self.assertEqual(statuses.count(200), 8, statuses)

        bodies = [r.json() for r in responses]
        winners = [b for b in bodies if b["created"]]
        self.assertEqual(len(winners), 1)
        winner_refs = _task_map(winners[0])
        self.assertEqual(set(winner_refs), set(sids))
        for b in bodies:
            refs = _task_map(b)
            # Every response names the same three task ids, one per subscription.
            self.assertEqual(set(refs), set(sids))
            self.assertEqual(
                {sid: ref["task_id"] for sid, ref in refs.items()},
                {sid: ref["task_id"] for sid, ref in winner_refs.items()},
            )
            for sid, ref in refs.items():
                # The first publication's schedule is never rewritten.
                self.assertEqual(ref["deliver_at"], winner_refs[sid]["deliver_at"])

        status = self.client.get(f"/events/{event_id}").json()
        self.assertEqual(status["counts"]["pending"], 3)
        self.assertEqual(status["counts"]["delivered"], 0)
        self.assertEqual(
            sorted(t["subscription_id"] for t in status["targets"]), sorted(sids)
        )

    def test_concurrent_replay_does_not_rewrite_first_deliver_at(self):
        sid = self._create_sub()
        event_id = "evt-schedule"
        barrier = threading.Barrier(2)
        outcomes = {}

        def first():
            barrier.wait()
            outcomes["first"] = self._submit(
                event_id, {"n": 1}, [sid], deliver_at=FUTURE
            )

        def replay():
            barrier.wait()
            outcomes["replay"] = self._submit(
                event_id, {"n": 1}, [sid], deliver_at=FUTURE2
            )

        threads = [threading.Thread(target=first), threading.Thread(target=replay)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        first = outcomes["first"]
        replay = outcomes["replay"]
        self.assertIn(first.status_code, (200, 201))
        self.assertIn(replay.status_code, (200, 201))
        self.assertEqual(first.status_code + replay.status_code, 401)  # 201 + 200

        first_ref = _all_task_refs(first.json())[0]
        replay_ref = _all_task_refs(replay.json())[0]
        self.assertEqual(first_ref["task_id"], replay_ref["task_id"])
        # Both responses report the single persisted task, whose deliver_at is
        # whichever request won -- identical for both, never the loser's time.
        self.assertEqual(first_ref["deliver_at"], replay_ref["deliver_at"])

        detail = self.client.get(f"/events/{event_id}").json()
        self.assertEqual(len(detail["targets"]), 1)
        self.assertEqual(detail["targets"][0]["deliver_at"], first_ref["deliver_at"])

    # ------------------------------------------------------------------
    # same id, different payload
    # ------------------------------------------------------------------
    def test_concurrent_different_payloads_one_winner_rest_409(self):
        sids = [self._create_sub() for _ in range(2)]
        event_id = "evt-payload-conflict"

        def fire(i):
            return self._submit(event_id, {"v": i}, sids, deliver_at=FUTURE)

        with ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(pool.map(fire, range(8)))

        statuses = [r.status_code for r in responses]
        self.assertNotIn(500, statuses, [r.text for r in responses])
        self.assertEqual(statuses.count(201), 1, statuses)
        self.assertEqual(statuses.count(409), 7, statuses)

        # Exactly the winner's two tasks exist; the losers persisted nothing.
        detail = self.client.get(f"/events/{event_id}")
        self.assertEqual(detail.status_code, 200)
        data = detail.json()
        self.assertEqual(len(data["targets"]), 2)
        self.assertEqual(
            sorted(t["subscription_id"] for t in data["targets"]), sorted(sids)
        )

    # ------------------------------------------------------------------
    # direct vs topic channel race
    # ------------------------------------------------------------------
    def test_concurrent_direct_vs_topic_is_stable_and_not_mixed(self):
        # Run multiple rounds with fresh ids so both possible winners
        # (direct-first and topic-first) are exercised.
        for round_no in range(8):
            direct_sids = [self._create_sub() for _ in range(2)]
            topic_sids = [self._create_sub() for _ in range(2)]
            topic = f"news-{round_no}"
            self._topic_members(topic, topic_sids)
            event_id = f"evt-channel-{round_no}"
            payload = {"round": round_no}
            barrier = threading.Barrier(8)

            def direct(_):
                barrier.wait()
                return self._submit(event_id, payload, direct_sids)

            def publish(_):
                barrier.wait()
                return self.client.post(
                    "/topic-events",
                    json={"event_id": event_id, "topic": topic, "payload": payload},
                )

            # Submit all eight callables before awaiting any of them: two
            # separate pool.map() calls would deadlock the barrier (waiting on
            # the first list(...) before submitting the second group).
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = (
                    [pool.submit(direct, i) for i in range(4)]
                    + [pool.submit(publish, i) for i in range(4)]
                )
                raw = [f.result() for f in futures]
            d_responses = raw[:4]
            t_responses = raw[4:]

            d_statuses = [r.status_code for r in d_responses]
            t_statuses = [r.status_code for r in t_responses]
            self.assertNotIn(500, d_statuses + t_statuses)

            winners_direct = [s for s in d_statuses if s in (200, 201)]
            winners_topic = [s for s in t_statuses if s in (200, 201)]
            self.assertEqual(
                len(winners_direct) + len(winners_topic), 4, (d_statuses, t_statuses)
            )

            detail = self.client.get(f"/events/{event_id}")
            self.assertEqual(detail.status_code, 200)
            recipients = {t["subscription_id"] for t in detail.json()["targets"]}

            if winners_direct:
                # Direct won: exactly one 201 from /events, all topic calls 409.
                self.assertEqual(d_statuses.count(201), 1, d_statuses)
                self.assertEqual(d_statuses.count(200), 3, d_statuses)
                self.assertEqual(set(t_statuses), {409}, t_statuses)
                self.assertEqual(recipients, set(direct_sids))
                self.assertTrue(recipients.isdisjoint(topic_sids))
            else:
                # Topic won: exactly one 201 from /topic-events, all direct
                # calls 409, and recipients are only the topic's members.
                self.assertEqual(set(d_statuses), {409}, d_statuses)
                self.assertEqual(t_statuses.count(201), 1, t_statuses)
                self.assertEqual(t_statuses.count(200), 3, t_statuses)
                self.assertEqual(recipients, set(topic_sids))
                self.assertTrue(recipients.isdisjoint(direct_sids))

    # ------------------------------------------------------------------
    # sequential status contract (unchanged response structure/semantics)
    # ------------------------------------------------------------------
    def test_exact_sequential_replay_returns_duplicates(self):
        sids = [self._create_sub(), self._create_sub()]
        first = self._submit("evt-seq", {"n": 1}, sids, deliver_at=FUTURE)
        self.assertEqual(first.status_code, 201)
        self.assertEqual(len(first.json()["created"]), 2)
        self.assertEqual(first.json()["duplicates"], [])

        second = self._submit("evt-seq", {"n": 1}, sids, deliver_at=FUTURE)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["created"], [])
        dupes = _task_map(second.json())
        winners = _task_map(first.json())
        self.assertEqual(
            {sid: ref["task_id"] for sid, ref in dupes.items()},
            {sid: ref["task_id"] for sid, ref in winners.items()},
        )

        # A replay naming an additional subscription creates only that one
        # task -> mixed created/duplicates -> 201.
        extra = self._create_sub()
        third = self._submit("evt-seq", {"n": 1}, sids + [extra], deliver_at=FUTURE2)
        self.assertEqual(third.status_code, 201)
        self.assertEqual([r["subscription_id"] for r in third.json()["created"]], [extra])
        self.assertEqual(len(third.json()["duplicates"]), 2)
        # The original tasks keep the original schedule, not FUTURE2.
        for ref in third.json()["duplicates"]:
            self.assertEqual(
                ref["deliver_at"], winners[ref["subscription_id"]]["deliver_at"]
            )

    def test_unknown_subscription_is_400_and_persists_nothing(self):
        resp = self._submit("evt-unknown", {"n": 1}, ["deadbeef0000"])
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(self.client.get("/events/evt-unknown").status_code, 404)

    def test_different_payload_sequential_is_409_and_persists_nothing_new(self):
        sid = self._create_sub()
        first = self._submit("evt-diff", {"n": 1}, [sid], deliver_at=FUTURE)
        self.assertEqual(first.status_code, 201)
        conflict = self._submit("evt-diff", {"n": 2}, [sid], deliver_at=FUTURE)
        self.assertEqual(conflict.status_code, 409)
        detail = self.client.get("/events/evt-diff").json()
        self.assertEqual(len(detail["targets"]), 1)

    def test_topic_id_collision_is_409(self):
        sid = self._create_sub()
        self._topic_members("t", [sid])
        topic_resp = self.client.post(
            "/topic-events",
            json={"event_id": "evt-x", "topic": "t", "payload": {"n": 1}},
        )
        self.assertEqual(topic_resp.status_code, 201)
        direct = self._submit("evt-x", {"n": 1}, [sid])
        self.assertEqual(direct.status_code, 409)


class SubmitEventConcurrencyDatabaseTests(unittest.TestCase):
    """Races at the Database boundary: no HTTP, no workers, raw guarantees."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=_TMP_ROOT)
        self.path = os.path.join(self.tmp.name, "db.db")
        self.db = Database(self.path)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def _sub(self) -> str:
        return self.db.create_subscription(HOOK_URL, "s")["id"]

    def test_concurrent_identical_submissions_create_one_task_per_subscription(self):
        sids = [self._sub() for _ in range(4)]
        payload = json.dumps({"n": 1}, separators=(",", ":"))
        event_id = "evt-race-identical"
        barrier = threading.Barrier(12)
        results = []
        lock = threading.Lock()

        def fire():
            barrier.wait()
            status, receipt = self.db.submit_direct_event(
                event_id, payload, sids, max_attempts=4
            )
            with lock:
                results.append((status, receipt))

        threads = [threading.Thread(target=fire) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        statuses = [s for s, _ in results]
        self.assertEqual(statuses.count("created"), 1, statuses)
        self.assertEqual(statuses.count("duplicate"), 11, statuses)
        for _, receipt in results:
            self.assertEqual(len(receipt["created"]) + len(receipt["duplicates"]), 4)
        # One row per subscription, no duplicates.
        rows = self.db._conn.execute(
            "SELECT subscription_id FROM tasks WHERE event_id = ?", (event_id,)
        ).fetchall()
        self.assertEqual(sorted(r["subscription_id"] for r in rows), sorted(sids))

    def test_concurrent_different_payloads_leave_only_winner(self):
        sids = [self._sub(), self._sub()]
        event_id = "evt-race-payload"
        barrier = threading.Barrier(6)
        outcomes = []
        lock = threading.Lock()

        def fire(i):
            barrier.wait()
            status, result = self.db.submit_direct_event(
                event_id, json.dumps({"v": i}, separators=(",", ":")), sids, 4
            )
            with lock:
                outcomes.append((status, result))

        threads = [threading.Thread(target=fire, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        statuses = [s for s, _ in outcomes]
        self.assertEqual(statuses.count("created"), 1, statuses)
        self.assertEqual(statuses.count("conflict"), 5, statuses)
        self.assertTrue(all(r == "payload" for s, r in outcomes if s == "conflict"))
        rows = self.db._conn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE event_id = ?", (event_id,)
        ).fetchone()
        self.assertEqual(rows["n"], 2)  # only the winner's task set

    def test_concurrent_direct_vs_topic_never_mixes_recipients(self):
        direct_sids = [self._sub() for _ in range(3)]
        topic_sids = [self._sub() for _ in range(3)]
        for sid in topic_sids:
            self.db.set_subscription_topics(sid, ["news"])
        payload = json.dumps({"n": 1}, separators=(",", ":"))

        # Repeat with fresh ids so either side can win.
        for round_no in range(8):
            event_id = f"evt-race-channel-{round_no}"
            barrier = threading.Barrier(8)
            lock = threading.Lock()
            direct_statuses, topic_statuses = [], []

            def direct():
                barrier.wait()
                status, _ = self.db.submit_direct_event(
                    event_id, payload, direct_sids, 4
                )
                with lock:
                    direct_statuses.append(status)

            def topic():
                barrier.wait()
                status, _ = self.db.publish_topic_event(
                    event_id, "news", payload, 4
                )
                with lock:
                    topic_statuses.append(status)

            threads = (
                [threading.Thread(target=direct) for _ in range(4)]
                + [threading.Thread(target=topic) for _ in range(4)]
            )
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            rows = self.db._conn.execute(
                "SELECT subscription_id FROM tasks WHERE event_id = ?", (event_id,)
            ).fetchall()
            recipients = {r["subscription_id"] for r in rows}

            direct_created = direct_statuses.count("created")
            topic_created = topic_statuses.count("created")
            self.assertEqual(direct_created + topic_created, 1)
            if direct_created:
                self.assertEqual(direct_statuses.count("duplicate"), 3)
                self.assertTrue(all(s == "conflict" for s in topic_statuses))
                self.assertEqual(recipients, set(direct_sids))
                self.assertTrue(recipients.isdisjoint(topic_sids))
            else:
                self.assertTrue(all(s == "conflict" for s in direct_statuses))
                self.assertEqual(topic_statuses.count("duplicate"), 3)
                self.assertEqual(recipients, set(topic_sids))
                self.assertTrue(recipients.isdisjoint(direct_sids))

    def test_failed_transaction_leaves_no_event_and_no_tasks(self):
        known = self._sub()
        event_id = "evt-atomic-rollback"
        payload = json.dumps({"n": 1}, separators=(",", ":"))

        # Unknown subscription: no event, no tasks.
        status, unknown = self.db.submit_direct_event(
            event_id, payload, [known, "missing000000"], 4
        )
        self.assertEqual(status, "unknown")
        self.assertEqual(unknown, ["missing000000"])
        self.assertIsNone(
            self.db._conn.execute(
                "SELECT 1 FROM events WHERE id = ?", (event_id,)
            ).fetchone()
        )
        self.assertEqual(
            self.db._conn.execute(
                "SELECT COUNT(*) AS n FROM tasks WHERE event_id = ?", (event_id,)
            ).fetchone()["n"],
            0,
        )

        # A topic publication claims the id first: the direct event rolls back.
        other = self._sub()
        self.db.set_subscription_topics(other, ["t"])
        status, reason = self.db.publish_topic_event(
            event_id, "t", payload, 4
        )
        self.assertEqual(status, "created")
        status, reason = self.db.submit_direct_event(
            event_id, payload, [known], 4
        )
        self.assertEqual((status, reason), ("conflict", "channel"))
        rows = self.db._conn.execute(
            "SELECT subscription_id FROM tasks WHERE event_id = ?", (event_id,)
        ).fetchall()
        self.assertEqual([r["subscription_id"] for r in rows], [other])


if __name__ == "__main__":
    unittest.main()
