from dotenv import load_dotenv
import os
import argparse  # for command-line argument parsing
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer
import torch
from peft import PeftModel
from transformers import BitsAndBytesConfig

# Load environment variables (e.g., HF_TOKEN)
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


def generate_response(lora_path, prompt, image_path):
    """
    Generate model response given a text prompt and image.
    Args:
        lora_path (str): Path to LoRA fine-tuned checkpoint.
        prompt (str): User query (text prompt).
        image_path (str): Local path to input image.
    """

    # Load processor (handles multimodal text + image)
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct", token=HF_TOKEN)

    # Load base pretrained model
    model = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM-256M-Instruct",
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",  # faster inference
    ).to("cuda")

    # Load LoRA fine-tuned adapter (adds fine-tuning weights)
    model = PeftModel.from_pretrained(model, lora_path)
    model.to("cuda")

    # Prepare multimodal input (chat format)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "url": image_path},
            ]
        },
    ]

    # Tokenize and process inputs for the model
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,  # adds <|assistant|> marker before generation
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    # Generate output (non-sampling, deterministic)
    generated_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=64,  # limit the output length
    )

    # Decode token IDs to human-readable text
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return generated_texts[0]


if __name__ == "__main__":
    # Argument parser for CLI input
    parser = argparse.ArgumentParser(description="Run inference on SmolVLM with LoRA fine-tuned weights.")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA fine-tuned checkpoint.")
    parser.add_argument("--prompt", type=str, default="Describe the image in detail.", help="Question to ask the model (optional).")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image file.")

    args = parser.parse_args()

    # Run inference with provided arguments
    generated_texts = generate_response(args.lora_path, args.prompt, args.image_path)
    
    print("\n=== Model Response ===")
    print(generated_texts)
    print("======================\n")
