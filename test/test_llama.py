import requests
import base64
import time
import argparse
import os

def send_vlm_request(url, image_path, prompt="Describe the image in detail."):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image file not found: {image_path}")

    # Encode image to base64
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    # Prepare payload for Llama_cpp server
    payload = {
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

    print(f"🔹 Sending request to {url} ...")
    start = time.time()
    response = requests.post(url, json=payload)
    end = time.time()

    if response.status_code != 200:
        print(f"❌ Request failed: {response.status_code}")
        print(response.text)
        return

    # Parse response
    result = response.json()
    message = result['choices'][0]['message']['content']
    duration = end - start

    print("\n--- Model Response ---")
    print(f"Assistant: {message}")
    print(f"\n✅ Processed in {duration:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Send image + text prompt to a Llama_cpp server for vision-language inference."
    )

    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="URL of the Llama_cpp server (e.g., http://localhost:8080/v1/chat/completions)."
    )

    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the input image file."
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe the image in detail.",
        help="Prompt or question to ask the model (optional)."
    )

    args = parser.parse_args()
    send_vlm_request(args.url, args.image_path, args.prompt)
