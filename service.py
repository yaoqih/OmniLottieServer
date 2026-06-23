import base64
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Dict, Optional, Tuple

from PIL import Image
from pydantic import BaseModel, Field, model_validator


SYSTEM_PROMPT = "You are a Lottie animation expert."
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "OmniLottie/OmniLottie")
DEFAULT_PROCESSOR_PATH = os.environ.get("PROCESSOR_PATH", "Qwen/Qwen2.5-VL-3B-Instruct")
DEFAULT_HF_CACHE_ROOT = os.environ.get("HF_HOME", "/runpod-volume/huggingface-cache")

_MODEL = None
_PROCESSOR = None
_DEVICE = None
_LOADED_KEY: Optional[Tuple[str, str]] = None
_MODEL_LOCK = threading.Lock()


class PredictRequest(BaseModel):
    task_type: str
    text: Optional[str] = None
    image_base64: Optional[str] = None
    video_base64: Optional[str] = None
    model_path: str = Field(default=DEFAULT_MODEL_PATH)
    processor_path: str = Field(default=DEFAULT_PROCESSOR_PATH)
    max_tokens: int = Field(default=4096, ge=1, le=8192)
    do_sample: bool = False
    temperature: float = Field(default=0.9, gt=0.0, le=2.0)
    top_p: float = Field(default=0.25, gt=0.0, le=1.0)
    top_k: int = Field(default=5, ge=1, le=128)
    return_preview: bool = False

    @model_validator(mode="after")
    def validate_task_inputs(self) -> "PredictRequest":
        if self.task_type not in {"text-to-lottie", "image-to-lottie", "video-to-lottie"}:
            raise ValueError("task_type must be one of text-to-lottie, image-to-lottie, video-to-lottie")
        if self.task_type == "text-to-lottie" and not self.text:
            raise ValueError("text is required for text-to-lottie")
        if self.task_type == "image-to-lottie" and not self.image_base64:
            raise ValueError("image_base64 is required for image-to-lottie")
        if self.task_type == "video-to-lottie" and not self.video_base64:
            raise ValueError("video_base64 is required for video-to-lottie")
        return self


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


def build_dummy_response(task_type: str, payload: PredictRequest, elapsed_ms: int) -> Dict[str, Any]:
    lottie_json = build_dummy_lottie(payload.text, task_type)
    return {
        "status": "ok",
        "dummy": True,
        "task_type": task_type,
        "model_path": payload.model_path,
        "processor_path": payload.processor_path,
        "elapsed_ms": elapsed_ms,
        "parameters": {
            "max_tokens": payload.max_tokens,
            "do_sample": payload.do_sample,
            "temperature": payload.temperature,
            "top_p": payload.top_p,
            "top_k": payload.top_k,
        },
        "candidates": [{"index": 1, "lottie_json": lottie_json}],
        "primary_lottie": lottie_json,
        "primary_lottie_json": json.dumps(lottie_json, ensure_ascii=False, indent=2),
    }


def _b64_to_pil(b64_str: str) -> Image.Image:
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _write_temp_bytes(b64_str: str, suffix: str) -> str:
    raw = base64.b64decode(b64_str)
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return path


def resolve_cached_hf_path(repo_id: str, cache_root: Optional[str] = None) -> Optional[str]:
    if not repo_id or "/" not in repo_id:
        return None
    hub_root = Path(cache_root or DEFAULT_HF_CACHE_ROOT)
    if hub_root.name != "hub":
        hub_root = hub_root / "hub"
    model_dir = hub_root / f"models--{repo_id.replace('/', '--')}"
    snapshots_dir = model_dir / "snapshots"
    refs_dir = model_dir / "refs"
    if not snapshots_dir.exists():
        return None

    ref_name = os.environ.get("MODEL_REVISION", "main")
    ref_file = refs_dir / ref_name
    if ref_file.exists():
        snapshot_name = ref_file.read_text(encoding="utf-8").strip()
        snapshot_path = snapshots_dir / snapshot_name
        if snapshot_path.exists():
            return str(snapshot_path)

    snapshots = sorted([p for p in snapshots_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if snapshots:
        return str(snapshots[-1])
    return None


def resolve_model_source(model_path: str) -> str:
    if not model_path:
        raise ValueError("model_path is required")
    if os.path.exists(model_path):
        return model_path
    cached_path = resolve_cached_hf_path(model_path)
    if cached_path:
        return cached_path
    return model_path


def ensure_model_loaded(model_path: str, processor_path: str) -> Tuple[Any, Any, Any]:
    global _MODEL, _PROCESSOR, _DEVICE, _LOADED_KEY
    resolved_model_path = resolve_model_source(model_path)
    resolved_processor_path = resolve_model_source(processor_path)
    with _MODEL_LOCK:
        if _LOADED_KEY == (resolved_model_path, resolved_processor_path) and _MODEL is not None:
            return _MODEL, _PROCESSOR, _DEVICE

        import torch
        from transformers import AutoProcessor

        from decoder_hf import LottieDecoder

        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "xpu:0" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
        )
        model = LottieDecoder.from_pretrained(
            resolved_model_path,
            torch_dtype=torch.bfloat16 if hasattr(torch, "bfloat16") else None,
            trust_remote_code=True,
        )
        model = model.to(device).eval()
        processor = AutoProcessor.from_pretrained(
            resolved_processor_path,
            padding_side="left",
            trust_remote_code=True,
        )

        _MODEL = model
        _PROCESSOR = processor
        _DEVICE = device
        _LOADED_KEY = (resolved_model_path, resolved_processor_path)
        return _MODEL, _PROCESSOR, _DEVICE


def run_generation(request_model: PredictRequest) -> Dict[str, Any]:
    t0 = time.time()
    if os.environ.get("ENABLE_DUMMY", "true").lower() == "true":
        return build_dummy_response(request_model.task_type, request_model, int((time.time() - t0) * 1000))

    import inference_hf

    model, processor, device = ensure_model_loaded(request_model.model_path, request_model.processor_path)

    temp_paths = []
    try:
        if request_model.task_type == "text-to-lottie":
            messages = inference_hf.build_messages("text", text_prompt=request_model.text)
        elif request_model.task_type == "image-to-lottie":
            image = _b64_to_pil(request_model.image_base64 or "")
            image = image.resize((448, 448), Image.LANCZOS)
            messages = inference_hf.build_messages("image", text_prompt=request_model.text, image=image)
        else:
            video_path = _write_temp_bytes(request_model.video_base64 or "", ".mp4")
            temp_paths.append(video_path)
            frames = inference_hf.load_frames_from_video(video_path, num_frames=8)
            messages = inference_hf.build_messages("video", video_frames=frames)

        inputs = inference_hf.prepare_inference_input(processor, messages, device)
        generated_ids = inference_hf.generate_lottie(
            model=model,
            inputs=inputs,
            max_tokens=request_model.max_tokens,
            device=device,
            use_sampling=request_model.do_sample,
            temperature=request_model.temperature,
            top_p=request_model.top_p,
            top_k=request_model.top_k,
        )
        lottie_json = inference_hf.tokens_to_lottie_json(generated_ids)
        elapsed_ms = int((time.time() - t0) * 1000)
        return {
            "status": "ok",
            "dummy": False,
            "task_type": request_model.task_type,
            "model_path": request_model.model_path,
            "processor_path": request_model.processor_path,
            "elapsed_ms": elapsed_ms,
            "parameters": {
                "max_tokens": request_model.max_tokens,
                "do_sample": request_model.do_sample,
                "temperature": request_model.temperature,
                "top_p": request_model.top_p,
                "top_k": request_model.top_k,
            },
            "num_tokens": len(generated_ids),
            "candidates": [{"index": 1, "lottie_json": lottie_json}],
            "primary_lottie": lottie_json,
            "primary_lottie_json": json.dumps(lottie_json, ensure_ascii=False, indent=2),
        }
    finally:
        for path in temp_paths:
            try:
                os.remove(path)
            except OSError:
                pass
