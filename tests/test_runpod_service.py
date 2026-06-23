import json
import os
import sys
import types
import unittest
from unittest import mock

from service import PredictRequest, build_dummy_response, run_generation
from runpod_client import _parse_business_output


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self._old_dummy = os.environ.get("ENABLE_DUMMY")
        os.environ["ENABLE_DUMMY"] = "true"

    def tearDown(self):
        if self._old_dummy is None:
            os.environ.pop("ENABLE_DUMMY", None)
        else:
            os.environ["ENABLE_DUMMY"] = self._old_dummy

    def test_text_request_requires_text(self):
        with self.assertRaises(ValueError):
            PredictRequest(task_type="text-to-lottie")

    def test_dummy_text_generation_returns_primary_lottie(self):
        req = PredictRequest(task_type="text-to-lottie", text="A bouncing ball")
        result = run_generation(req)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["dummy"])
        self.assertEqual(result["task_type"], "text-to-lottie")
        self.assertIn("primary_lottie", result)
        self.assertIn("layers", result["primary_lottie"])

    def test_build_dummy_response_contains_serialized_json(self):
        req = PredictRequest(task_type="text-to-lottie", text="Spin")
        result = build_dummy_response(req.task_type, req, 12)
        parsed = json.loads(result["primary_lottie_json"])
        self.assertEqual(parsed["nm"], result["primary_lottie"]["nm"])


class HandlerTests(unittest.TestCase):
    def test_handler_rejects_missing_input(self):
        fake_runpod = types.SimpleNamespace(serverless=types.SimpleNamespace(start=lambda *_args, **_kwargs: None))
        with mock.patch.dict(sys.modules, {"runpod": fake_runpod}):
            import importlib
            import handler as handler_module

            importlib.reload(handler_module)
            result = handler_module.handler({})
        self.assertIn("error", result)


class ClientParseTests(unittest.TestCase):
    def test_parse_nested_output(self):
        payload = {
            "status": "COMPLETED",
            "output": {
                "status": "ok",
                "task_type": "text-to-lottie",
                "primary_lottie": {"nm": "Demo"},
                "elapsed_ms": 99,
                "candidates": [{"index": 1, "lottie_json": {"nm": "Demo"}}],
            },
        }
        parsed = _parse_business_output(payload)
        self.assertEqual(parsed["status"], "ok")
        self.assertEqual(parsed["task_type"], "text-to-lottie")
        self.assertEqual(parsed["primary_lottie"]["nm"], "Demo")


if __name__ == "__main__":
    unittest.main()
