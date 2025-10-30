import requests
import base64
import time
import argparse
import os

def send_vlm_request(url, model, image_path, prompt="Describe the image in detail."):
    # Kiểm tra ảnh tồn tại
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image file not found: {image_path}")

    # Mã hóa ảnh sang base64
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    # Payload gửi tới vLLM server
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1024,
        "temperature": 0,
    }

    print(f"🔹 Sending request to {url} using model: {model}")
    start = time.time()
    response = requests.post(url, json=payload)
    end = time.time()

    if response.status_code != 200:
        print(f"❌ Request failed ({response.status_code}):\n{response.text}")
        return

    result = response.json()
    message = result['choices'][0]['message']['content']
    duration = end - start

    print("\n--- Model Response ---")
    print(f"Assistant: {message}")
    print(f"\n✅ Processed in {duration:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Send an image and text prompt to a vLLM server using a specific model."
    )

    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="URL of the vLLM server (e.g., http://localhost:8000/v1/chat/completions)."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., SmolVLM-256M-Instruct)."
    )

    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the image file to analyze."
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe the image in detail.",
        help="Text prompt or question to ask the model (optional)."
    )

    args = parser.parse_args()
    send_vlm_request(args.url, args.model, args.image_path, args.prompt)
