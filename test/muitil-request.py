import asyncio
import aiohttp
import time
import json
import base64
import random
from glob import glob
import os
import argparse

async def send(session, url, payloads):
    """Send one random payload asynchronously."""
    payload = random.choice(payloads)
    async with session.post(url, json=payload, timeout=30) as resp:
        try:
            await resp.text()
        except Exception:
            pass  # Ignore decoding errors


async def main(args):
    url = args.url
    root_dir = args.root_dir
    host = args.host
    model = args.model
    prompt = args.prompt
    num_requests = args.num_requests
    concurrent = args.concurrent

    # Collect all image paths
    image_paths = glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True) \
            + glob(os.path.join(root_dir, "**", "*.jpeg"), recursive=True) \
            + glob(os.path.join(root_dir, "**", "*.png"), recursive=True)
    if not image_paths:
        raise RuntimeError(f"❌ No images found in directory: {root_dir}")

    # Prepare payloads
    payloads = []
    for _ in range(min(200, len(image_paths))):
        img_path = random.choice(image_paths)
        with open(img_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            }
        ]

        payload = {
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1024,
            "temperature": 0,
        }

        # Only add "model" field if using vLLM
        if host == "vllm":
            if not model:
                raise ValueError("❌ --model is required when host is 'vllm'")
            payload["model"] = model

        payloads.append(payload)

    print(f"✅ Prepared {len(payloads)} payloads with {len(image_paths)} available images.")
    print(f"🔹 Sending {num_requests} requests ({concurrent} concurrent) to {host} host: {url}")

    # Run benchmark
    start = time.time()
    async with aiohttp.ClientSession() as session:
        for i in range(0, num_requests, concurrent):
            tasks = [send(session, url, payloads) for _ in range(concurrent)]
            await asyncio.gather(*tasks)
    end = time.time()

    rpm = num_requests / (end - start) * 60
    print(f"\n✅ Throughput: {rpm:.2f} requests per minute ({end - start:.2f} seconds total)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Asynchronous benchmark script for vLLM or llama.cpp servers with image input."
    )

    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="URL of the server endpoint (e.g., http://localhost:8000/v1/chat/completions)."
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Directory containing image files."
    )

    parser.add_argument(
        "--host",
        type=str,
        choices=["vllm", "llama_cpp"],
        required=True,
        help="Server backend type: 'vllm' or 'llama_cpp'."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (required only if host == 'vllm')."
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe the image in detail.",
        help="Prompt or question to send with the image (optional)."
    )

    parser.add_argument(
        "--num_requests",
        type=int,
        default=50,
        help="Total number of requests to send (default: 50)."
    )

    parser.add_argument(
        "--concurrent",
        type=int,
        default=5,
        help="Number of concurrent requests (default: 5)."
    )

    args = parser.parse_args()
    asyncio.run(main(args))
