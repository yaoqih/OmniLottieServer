import json
import os
import time
from typing import Any, Dict, Optional

import runpod


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def build_dummy_lottie(text: Optional[str], task_type: str) -> Dict[str, Any]:
    label = (text or task_type or "OmniLottie").strip()[:48]
    return {
        "v": "5.5.2",
        "fr": 8,
        "ip": 0,
        "op": 24,
        "w": 512,
        "h": 512,
        "nm": f"Dummy {task_type}",
        "ddd": 0,
        "assets": [],
        "layers": [
            {
                "ddd": 0,
                "ind": 1,
                "ty": 4,
                "nm": label or "Dummy layer",
                "sr": 1,
                "ks": {
                    "o": {"a": 0, "k": 100},
                    "r": {"a": 0, "k": 0},
                    "p": {"a": 0, "k": [256, 256, 0]},
                    "a": {"a": 0, "k": [0, 0, 0]},
                    "s": {"a": 0, "k": [100, 100, 100]},
                },
                "shapes": [
                    {
                        "ty": "el",
                        "p": {"a": 0, "k": [0, 0]},
                        "s": {
                            "a": 1,
                            "k": [
                                {"t": 0, "s": [120, 120]},
                                {"t": 12, "s": [180, 180]},
                                {"t": 24, "s": [120, 120]},
                            ],
                        },
                        "nm": "Pulse",
                    },
                    {
                        "ty": "fl",
                        "c": {"a": 0, "k": [0.145, 0.529, 0.91, 1]},
                        "o": {"a": 0, "k": 100},
                        "r": 1,
                        "nm": "Fill",
                    },
                ],
                "ip": 0,
                "op": 24,
                "st": 0,
                "bm": 0,
            }
        ],
    }


def build_dummy_response(task_type: str, payload: Dict[str, Any], elapsed_ms: int) -> Dict[str, Any]:
    lottie_json = build_dummy_lottie(payload.get("text"), task_type)
    return {
        "status": "ok",
        "dummy": True,
        "task_type": task_type,
        "model_path": payload.get("model_path") or os.environ.get("MODEL_PATH", "OmniLottie/OmniLottie"),
        "processor_path": payload.get("processor_path") or os.environ.get("PROCESSOR_PATH", "Qwen/Qwen2.5-VL-3B-Instruct"),
        "elapsed_ms": elapsed_ms,
        "parameters": {
            "max_tokens": payload.get("max_tokens", 4096),
            "do_sample": bool(payload.get("do_sample", False)),
            "temperature": payload.get("temperature", 0.9),
            "top_p": payload.get("top_p", 0.25),
            "top_k": payload.get("top_k", 5),
        },
        "candidates": [{"index": 1, "lottie_json": lottie_json}],
        "primary_lottie": lottie_json,
        "primary_lottie_json": json.dumps(lottie_json, ensure_ascii=False, indent=2),
    }


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    try:
        payload = event.get("input") if isinstance(event, dict) else None
        if not isinstance(payload, dict):
            return {"error": "Missing or invalid 'input' payload."}

        task_type = str(payload.get("task_type", "")).strip()
        if task_type not in {"text-to-lottie", "image-to-lottie", "video-to-lottie"}:
            return {"error": "task_type must be one of text-to-lottie, image-to-lottie, video-to-lottie"}

        if os.environ.get("ENABLE_DUMMY", "true").lower() == "true":
            return build_dummy_response(task_type, payload, int((time.time() - t0) * 1000))

        from service import PredictRequest, run_generation

        request_model = PredictRequest(**payload)
        result = run_generation(request_model)
        result["elapsed_ms"] = int((time.time() - t0) * 1000)
        return result
    except Exception as exc:
        return {"error": str(exc)}


runpod.serverless.start({"handler": handler})
