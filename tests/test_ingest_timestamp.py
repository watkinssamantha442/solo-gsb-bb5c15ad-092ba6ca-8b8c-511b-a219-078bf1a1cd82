"""Regression tests: POST /ingest accepts only finite X-Webhook-Timestamp values.

The timestamp window check used to be ``ts = float(header)`` followed by
``abs(time.time() - ts) > tolerance``. ``float()`` also accepts the strings
``NaN`` and ``Infinity`` (and overflowing literals such as ``1e400`` parse to
``inf`` without raising). A ``NaN`` timestamp made every skew comparison
``False``, so a correctly signed request with ``X-Webhook-Timestamp: NaN``
sailed past the replay window, verified, and was persisted.

These tests pin the fixed contract:

- ``NaN`` (any case), ``+/-Infinity``/``inf`` and overflow values (``1e400``,
  huge digit strings) are rejected with ``400`` -- even when the signature
  over the exact header text is valid -- and nothing is persisted;
- unparseable timestamps keep returning ``400``;
- the +/-300s skew window is enforced with inclusive boundaries (exactly
  +/-300s passes, anything beyond fails with ``401``);
- signature compatibility is preserved: the HMAC covers the *original header
  text*, so finite but non-canonical spellings (``+1800000000``,
  ``1.8000e+09``, fractional seconds) verify when signed as sent, and a
  signature over a normalized spelling does not;
- the happy path (valid signature, current timestamp) still returns
  ``201``/``200`` with the idempotent replay contract intact.

Pure stdlib + the app's own FastAPI test client; run with either of

    python3 -m unittest discover -s tests -v
    python3 -m pytest tests/
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
import unittest
from unittest import mock

# A private temp database must be selected before app.config (and hence
# app.main) is first imported, since config reads the environment at import.
_TMP_ROOT = tempfile.mkdtemp(prefix="wr-ingest-ts-")
os.environ.setdefault("DB_PATH", os.path.join(_TMP_ROOT, "default.db"))
os.environ.setdefault("INGEST_SECRET", "test-ingest-secret")
# Delivery targets never accept a connection (port 1); background workers
# only move the created tasks through their normal retry lifecycle.
os.environ.setdefault("WORKER_COUNT", "1")

from fastapi.testclient import TestClient  # noqa: E402

from app import config  # noqa: E402
import app.main as main  # noqa: E402

SECRET = "test-ingest-secret"
HOOK_URL = "http://127.0.0.1:1/hook"
# Fixed wall clock for the skew-window boundary tests.
NOW = 1_800_000_000.0
TOLERANCE = 300.0  # pinned in setUp via config.INGEST_TIMESTAMP_TOLERANCE


def _sign(timestamp: str, webhook_id: str, body: bytes) -> str:
    """Documented inbound signature: sha256=<HMAC(secret, ts + "." + id + "." + body)>."""
    signed = timestamp.encode("utf-8") + b"." + webhook_id.encode("utf-8") + b"." + body
    return "sha256=" + hmac.new(SECRET.encode("utf-8"), signed, hashlib.sha256).hexdigest()


class _FrozenTime:
    """Stand-in for the ``time`` module with a fixed wall clock."""

    def __init__(self, now: float):
        self._now = now

    def time(self) -> float:
        return self._now


class IngestTimestampTests(unittest.TestCase):
    """End-to-end HTTP tests through the real FastAPI lifespan."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=_TMP_ROOT)
        # config reads the environment once at import; under unittest
        # discovery another test module may have triggered that import, so
        # pin the ingest-relevant settings explicitly for this test case.
        self._saved_config = (config.INGEST_SECRET, config.INGEST_TIMESTAMP_TOLERANCE)
        config.INGEST_SECRET = SECRET
        config.INGEST_TIMESTAMP_TOLERANCE = TOLERANCE
        config.DB_PATH = os.path.join(self.tmp.name, "api.db")
        self.client = TestClient(main.app)
        self.client.__enter__()
        self.addCleanup(self._shutdown)
        self._seq = 0
        self.sid = self._create_sub()

    def _shutdown(self):
        self.client.__exit__(None, None, None)
        (config.INGEST_SECRET, config.INGEST_TIMESTAMP_TOLERANCE) = self._saved_config
        self.tmp.cleanup()

    def _create_sub(self) -> str:
        resp = self.client.post(
            "/subscriptions", json={"target_url": HOOK_URL, "secret": "s3cr3t"}
        )
        self.assertEqual(resp.status_code, 201, resp.text)
        return resp.json()["id"]

    def _next_id(self) -> str:
        self._seq += 1
        return f"evt-ts-{self._seq}"

    def _body(self) -> bytes:
        return json.dumps(
            {"subscription_ids": [self.sid], "type": "order.paid", "order_id": 42},
            separators=(",", ":"),
        ).encode("utf-8")

    def _ingest(
        self,
        timestamp: str,
        webhook_id: str,
        body: bytes,
        sign_over: str | None = None,
    ):
        """POST /ingest with a valid signature over ``sign_over`` (default: the header text)."""
        return self.client.post(
            "/ingest",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Webhook-Timestamp": timestamp,
                "X-Webhook-Id": webhook_id,
                "X-Webhook-Signature": _sign(
                    timestamp if sign_over is None else sign_over, webhook_id, body
                ),
            },
        )

    # ------------------------------------------------------------------
    # non-finite timestamps: rejected before the skew check and before
    # signature verification can be reached, even with a valid signature
    # ------------------------------------------------------------------
    def test_non_finite_timestamps_are_400_even_when_correctly_signed(self):
        non_finite = [
            "NaN", "nan", "NAN", "+NaN",
            "Infinity", "inf", "+Infinity", "-Infinity", "-inf",
            "1e400", "-1e400",  # overflow parses to +/-inf without raising
            "1" + "0" * 400,    # huge integer literal overflows to inf
        ]
        for ts_text in non_finite:
            with self.subTest(timestamp=ts_text):
                webhook_id = self._next_id()
                resp = self._ingest(ts_text, webhook_id, self._body())
                self.assertEqual(resp.status_code, 400, resp.text)
                self.assertIn("finite", resp.json()["detail"])
                # Nothing was persisted under that webhook id.
                self.assertEqual(self.client.get(f"/events/{webhook_id}").status_code, 404)

    def test_signed_nan_no_longer_bypasses_the_window(self):
        # The original bug: float("NaN") succeeded and abs(now - nan) > 300
        # was False, so this exact request verified and was stored (201).
        webhook_id = self._next_id()
        resp = self._ingest("NaN", webhook_id, self._body())
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertEqual(self.client.get(f"/events/{webhook_id}").status_code, 404)

    def test_unparseable_timestamps_remain_400(self):
        for ts_text in ["abc", "1.2.3", "0x10", "12:34", "2026-09-20T00:00:00Z"]:
            with self.subTest(timestamp=ts_text):
                webhook_id = self._next_id()
                resp = self._ingest(ts_text, webhook_id, self._body())
                self.assertEqual(resp.status_code, 400, resp.text)
                self.assertEqual(self.client.get(f"/events/{webhook_id}").status_code, 404)

    # ------------------------------------------------------------------
    # skew window: inclusive +/-300s boundaries around a frozen clock
    # ------------------------------------------------------------------
    def test_skew_window_boundaries_are_inclusive(self):
        cases = [
            (NOW, 201),                              # no skew
            (NOW - TOLERANCE, 201),                  # exactly at the lower edge
            (NOW + TOLERANCE, 201),                  # exactly at the upper edge
            (NOW - TOLERANCE - 0.5, 401),            # just outside
            (NOW + TOLERANCE + 0.5, 401),            # just outside
            (NOW - TOLERANCE - 1, 401),
            (NOW + TOLERANCE + 1, 401),
        ]
        with mock.patch.object(main, "time", _FrozenTime(NOW)):
            for ts_value, expected in cases:
                with self.subTest(timestamp=ts_value, expected=expected):
                    webhook_id = self._next_id()
                    resp = self._ingest(repr(ts_value), webhook_id, self._body())
                    self.assertEqual(resp.status_code, expected, resp.text)
                    persisted = self.client.get(f"/events/{webhook_id}").status_code
                    self.assertEqual(persisted, 200 if expected == 201 else 404)

    # ------------------------------------------------------------------
    # signature compatibility: the HMAC covers the original header text
    # ------------------------------------------------------------------
    def test_finite_noncanonical_spellings_verify_as_sent(self):
        # Finite spellings whose float normalization differs from the raw
        # text; signing the raw header text must succeed (no normalization).
        spellings = [
            f"+{NOW:.0f}",        # leading plus
            f"{NOW:.4e}",         # scientific notation
            f"{NOW:.3f}",         # fractional seconds
            f"{NOW:.1f}",         # trailing .0
        ]
        with mock.patch.object(main, "time", _FrozenTime(NOW)):
            for ts_text in spellings:
                with self.subTest(timestamp=ts_text):
                    webhook_id = self._next_id()
                    resp = self._ingest(ts_text, webhook_id, self._body())
                    self.assertEqual(resp.status_code, 201, resp.text)

    def test_signature_over_normalized_spelling_does_not_verify(self):
        # The server must not normalize before verifying: a signature over
        # str(float(header)) instead of the raw header text is a mismatch.
        ts_text = f"+{NOW:.0f}"
        normalized = str(float(ts_text))  # "1800000000.0" != "+1800000000"
        self.assertNotEqual(ts_text, normalized)
        with mock.patch.object(main, "time", _FrozenTime(NOW)):
            resp = self._ingest(
                ts_text, self._next_id(), self._body(), sign_over=normalized
            )
        self.assertEqual(resp.status_code, 401, resp.text)

    # ------------------------------------------------------------------
    # happy path and idempotent replay are unchanged
    # ------------------------------------------------------------------
    def test_valid_request_ingests_and_replays_idempotently(self):
        webhook_id = self._next_id()
        body = self._body()
        with mock.patch.object(main, "time", _FrozenTime(NOW)):
            first = self._ingest(f"{NOW:.0f}", webhook_id, body)
            self.assertEqual(first.status_code, 201, first.text)
            created = first.json()["created"]
            self.assertEqual(len(created), 1)
            self.assertEqual(created[0]["subscription_id"], self.sid)

            # Exact replay: same id, same body, fresh in-window timestamp.
            # The stored first receipt is returned verbatim with 200.
            replay = self._ingest(f"{NOW + 1:.0f}", webhook_id, body)
            self.assertEqual(replay.status_code, 200, replay.text)
            self.assertEqual(replay.json(), first.json())

        status = self.client.get(f"/events/{webhook_id}")
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["event_id"], webhook_id)

    def test_rejected_timestamp_does_not_claim_the_webhook_id(self):
        # A 400 rejection persists nothing, so the same id remains usable
        # for a later valid request (no phantom idempotency/conflict).
        webhook_id = self._next_id()
        body = self._body()
        with mock.patch.object(main, "time", _FrozenTime(NOW)):
            bad = self._ingest("NaN", webhook_id, body)
            self.assertEqual(bad.status_code, 400, bad.text)
            good = self._ingest(f"{NOW:.0f}", webhook_id, body)
            self.assertEqual(good.status_code, 201, good.text)
            self.assertEqual(len(good.json()["created"]), 1)


if __name__ == "__main__":
    unittest.main()
