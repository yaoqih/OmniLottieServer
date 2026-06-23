import os
import time
from typing import Any, Dict

import runpod


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    try:
        payload = event.get("input") if isinstance(event, dict) else None
        if not isinstance(payload, dict):
            return {"error": "Missing or invalid 'input' payload."}

        from service import PredictRequest, run_generation

        request_model = PredictRequest(**payload)
        result = run_generation(request_model)
        result["elapsed_ms"] = int((time.time() - t0) * 1000)
        return result
    except Exception as exc:
        return {"error": str(exc)}


runpod.serverless.start({"handler": handler})
