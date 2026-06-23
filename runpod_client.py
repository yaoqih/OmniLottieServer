import base64
import io
import os
import random
import time
from typing import Any, Dict, Optional, Tuple, Union

import requests


API_BASE = "https://api.runpod.ai/v2"


class RunpodClientError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None, raw: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.raw = raw


def _get_env() -> Tuple[str, str]:
    api_key = os.environ.get("RUNPOD_API_KEY")
    endpoint_id = os.environ.get("ENDPOINT_ID")
    if not api_key or not endpoint_id:
        raise RunpodClientError(
            "Missing RUNPOD_API_KEY or ENDPOINT_ID",
            raw={"RUNPOD_API_KEY": bool(api_key), "ENDPOINT_ID": bool(endpoint_id)},
        )
    return api_key, endpoint_id


def _headers(api_key: str) -> Dict[str, str]:
    return {
        "accept": "application/json",
        "authorization": api_key,
        "content-type": "application/json",
    }


def encode_file_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_image_to_base64(image: Union["Image.Image", "np.ndarray"]) -> str:
    try:
        from PIL import Image
        import numpy as np
    except Exception as exc:
        raise RunpodClientError("Pillow and numpy are required for image encoding", raw=str(exc))

    if isinstance(image, Image.Image):
        img = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert("RGB")
    else:
        raise RunpodClientError("Unsupported image type", raw=str(type(image)))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_business_output(obj: Dict[str, Any]) -> Dict[str, Any]:
    data = obj or {}
    out = data.get("output") if isinstance(data.get("output"), dict) else data
    candidates = out.get("candidates") if isinstance(out.get("candidates"), list) else []
    first_candidate = candidates[0] if candidates else {}
    primary = out.get("primary_lottie") or first_candidate.get("lottie_json")
    primary_json = out.get("primary_lottie_json")
    if primary_json is None and primary is not None:
        import json

        primary_json = json.dumps(primary, ensure_ascii=False, indent=2)

    return {
        "status": out.get("status") or data.get("status"),
        "job_status": data.get("status"),
        "task_type": out.get("task_type"),
        "model_path": out.get("model_path"),
        "processor_path": out.get("processor_path"),
        "elapsed_ms": out.get("elapsed_ms"),
        "num_tokens": out.get("num_tokens"),
        "candidates": candidates,
        "primary_lottie": primary,
        "primary_lottie_json": primary_json,
        "delayTime": data.get("delayTime"),
        "executionTime": data.get("executionTime"),
        "dummy": out.get("dummy"),
    }


def _build_input_payload(
    task_type: str,
    *,
    text: Optional[str] = None,
    image_base64: Optional[str] = None,
    video_base64: Optional[str] = None,
    model_path: Optional[str] = None,
    processor_path: Optional[str] = None,
    max_tokens: Optional[int] = None,
    do_sample: Optional[bool] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    return_preview: Optional[bool] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"task_type": task_type}
    for key, value in {
        "text": text,
        "image_base64": image_base64,
        "video_base64": video_base64,
        "model_path": model_path,
        "processor_path": processor_path,
        "max_tokens": max_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "return_preview": return_preview,
    }.items():
        if value is not None:
            payload[key] = value
    return {"input": payload}


def runsync(
    task_type: str,
    *,
    text: Optional[str] = None,
    image_base64: Optional[str] = None,
    video_base64: Optional[str] = None,
    model_path: Optional[str] = None,
    processor_path: Optional[str] = None,
    max_tokens: Optional[int] = None,
    do_sample: Optional[bool] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    return_preview: bool = False,
    wait_ms: int = 120000,
) -> Dict[str, Any]:
    api_key, endpoint_id = _get_env()
    url = f"{API_BASE}/{endpoint_id}/runsync?wait={wait_ms}"
    payload = _build_input_payload(
        task_type,
        text=text,
        image_base64=image_base64,
        video_base64=video_base64,
        model_path=model_path,
        processor_path=processor_path,
        max_tokens=max_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        return_preview=return_preview,
    )
    try:
        resp = requests.post(url, headers=_headers(api_key), json=payload, timeout=(10, max(wait_ms / 1000, 30)))
        if resp.status_code != 200:
            raise RunpodClientError(f"HTTP {resp.status_code}: runsync failed", resp.status_code, resp.text[:500])
        data = resp.json()
        if isinstance(data, dict) and "error" in data:
            raise RunpodClientError(f"Service error: {data['error']}", resp.status_code, data)
        return _parse_business_output(data)
    except requests.RequestException as exc:
        raise RunpodClientError("Network error talking to RunPod", raw=str(exc))


def run_async(
    task_type: str,
    *,
    text: Optional[str] = None,
    image_base64: Optional[str] = None,
    video_base64: Optional[str] = None,
    model_path: Optional[str] = None,
    processor_path: Optional[str] = None,
    max_tokens: Optional[int] = None,
    do_sample: Optional[bool] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    return_preview: bool = False,
    timeout_s: int = 180,
    base_interval_s: float = 2.0,
) -> Dict[str, Any]:
    api_key, endpoint_id = _get_env()
    payload = _build_input_payload(
        task_type,
        text=text,
        image_base64=image_base64,
        video_base64=video_base64,
        model_path=model_path,
        processor_path=processor_path,
        max_tokens=max_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        return_preview=return_preview,
    )
    session = requests.Session()
    submit = session.post(f"{API_BASE}/{endpoint_id}/run", headers=_headers(api_key), json=payload, timeout=(10, 30))
    if submit.status_code != 200:
        raise RunpodClientError(f"HTTP {submit.status_code}: submit failed", submit.status_code, submit.text[:500])
    data = submit.json()
    job_id = data.get("id") or data.get("jobId") or data.get("job_id")
    if not job_id:
        raise RunpodClientError("Missing job id", raw=data)

    t0 = time.time()
    backoff = 1.0
    status_url = f"{API_BASE}/{endpoint_id}/status/{job_id}"
    while True:
        if time.time() - t0 > timeout_s:
            raise RunpodClientError("Polling timed out", raw={"job_id": job_id})
        resp = session.get(status_url, headers=_headers(api_key), timeout=(5, 30))
        if resp.status_code == 429:
            time.sleep(backoff + random.uniform(0, 0.5))
            backoff = min(backoff * 2, 16.0)
            continue
        if resp.status_code != 200:
            time.sleep(base_interval_s)
            continue
        data = resp.json()
        status = str(data.get("status", "")).upper()
        if data.get("error"):
            raise RunpodClientError(f"Service error: {data['error']}", resp.status_code, data)
        if status in {"COMPLETED", "SUCCESS", "FINISHED"}:
            return _parse_business_output(data)
        if status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
            raise RunpodClientError(f"Job ended with status {status}", resp.status_code, data)
        time.sleep(base_interval_s)


def ensure_env_ready() -> Tuple[str, str]:
    return _get_env()
