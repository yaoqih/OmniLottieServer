import argparse
import json

import gradio as gr
from dotenv import load_dotenv

from runpod_client import RunpodClientError, encode_file_to_base64, encode_image_to_base64, runsync


load_dotenv()


def _status_text(result: dict) -> str:
    return "\n".join(
        [
            f"status: {result.get('status')}",
            f"task_type: {result.get('task_type')}",
            f"model_path: {result.get('model_path')}",
            f"elapsed_ms: {result.get('elapsed_ms')}",
            f"num_tokens: {result.get('num_tokens')}",
            f"dummy: {result.get('dummy')}",
        ]
    )


def submit_text(text, model_path, processor_path, max_tokens, do_sample, temperature, top_p, top_k):
    result = runsync(
        "text-to-lottie",
        text=text,
        model_path=model_path or None,
        processor_path=processor_path or None,
        max_tokens=max_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    return result.get("primary_lottie_json"), _status_text(result), json.dumps(result.get("primary_lottie"), ensure_ascii=False, indent=2)


def submit_image(image, text, model_path, processor_path, max_tokens, do_sample, temperature, top_p, top_k):
    image_base64 = encode_image_to_base64(image) if image is not None else None
    result = runsync(
        "image-to-lottie",
        text=text,
        image_base64=image_base64,
        model_path=model_path or None,
        processor_path=processor_path or None,
        max_tokens=max_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    return result.get("primary_lottie_json"), _status_text(result), json.dumps(result.get("primary_lottie"), ensure_ascii=False, indent=2)


def submit_video(video, model_path, processor_path, max_tokens, do_sample, temperature, top_p, top_k):
    video_base64 = encode_file_to_base64(video) if video else None
    result = runsync(
        "video-to-lottie",
        video_base64=video_base64,
        model_path=model_path or None,
        processor_path=processor_path or None,
        max_tokens=max_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    return result.get("primary_lottie_json"), _status_text(result), json.dumps(result.get("primary_lottie"), ensure_ascii=False, indent=2)


with gr.Blocks(title="OmniLottie RunPod Client") as demo:
    gr.Markdown("# OmniLottie RunPod Client")
    gr.Markdown("Thin Gradio frontend for a RunPod Serverless OmniLottie endpoint.")
    with gr.Accordion("Generation Settings", open=True):
        model_path = gr.Textbox(label="Model Path", value="OmniLottie/OmniLottie")
        processor_path = gr.Textbox(label="Processor Path", value="Qwen/Qwen2.5-VL-3B-Instruct")
        max_tokens = gr.Slider(label="Max Tokens", minimum=256, maximum=8192, step=256, value=4096)
        do_sample = gr.Checkbox(label="Use Sampling", value=False)
        temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, step=0.05, value=0.9)
        top_p = gr.Slider(label="Top P", minimum=0.05, maximum=1.0, step=0.05, value=0.25)
        top_k = gr.Slider(label="Top K", minimum=1, maximum=128, step=1, value=5)

    with gr.Tabs():
        with gr.Tab("Text-to-Lottie"):
            text_prompt = gr.Textbox(label="Prompt", lines=5)
            text_button = gr.Button("Generate")
            text_json = gr.Code(label="Primary Lottie JSON", language="json")
            text_status = gr.Textbox(label="Status")
            text_raw = gr.Code(label="Parsed Primary Object", language="json")
            text_button.click(
                submit_text,
                [text_prompt, model_path, processor_path, max_tokens, do_sample, temperature, top_p, top_k],
                [text_json, text_status, text_raw],
            )

        with gr.Tab("Image-to-Lottie"):
            image_input = gr.Image(label="Image", type="pil")
            image_text = gr.Textbox(label="Optional Text Prompt", lines=3)
            image_button = gr.Button("Generate")
            image_json = gr.Code(label="Primary Lottie JSON", language="json")
            image_status = gr.Textbox(label="Status")
            image_raw = gr.Code(label="Parsed Primary Object", language="json")
            image_button.click(
                submit_image,
                [image_input, image_text, model_path, processor_path, max_tokens, do_sample, temperature, top_p, top_k],
                [image_json, image_status, image_raw],
            )

        with gr.Tab("Video-to-Lottie"):
            video_input = gr.Video(label="Video")
            video_button = gr.Button("Generate")
            video_json = gr.Code(label="Primary Lottie JSON", language="json")
            video_status = gr.Textbox(label="Status")
            video_raw = gr.Code(label="Parsed Primary Object", language="json")
            video_button.click(
                submit_video,
                [video_input, model_path, processor_path, max_tokens, do_sample, temperature, top_p, top_k],
                [video_json, video_status, video_raw],
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    demo.launch(server_name=args.listen, server_port=args.port, share=args.share, show_error=args.debug)
