import os
import torch
import argparse
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer
from peft import PeftModel

def merge_lora_model(lora_path, output_path):
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")

    print(f"🔹 Loading base model and processor...")
    processor = AutoProcessor.from_pretrained(
        "HuggingFaceTB/SmolVLM-256M-Instruct",
        token=HF_TOKEN
    )
    model = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM-256M-Instruct",
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    print(f"🔹 Loading LoRA weights from: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)

    print("🔹 Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"🔹 Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct", token=HF_TOKEN)
    tokenizer.save_pretrained(output_path)
    processor.save_pretrained(output_path)

    print("✅ Merge and save completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge a LoRA checkpoint into the base SmolVLM model and save it."
    )

    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to the LoRA checkpoint directory."
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="./Merged_lora",
        help="Output directory for the merged model (default: ./Merged_lora)."
    )

    args = parser.parse_args()
    merge_lora_model(args.lora_path, args.output_path)
