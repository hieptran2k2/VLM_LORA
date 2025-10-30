from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from dotenv import load_dotenv
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from transformers import BitsAndBytesConfig
import time

from clear_memory import clear_memory  # custom helper to clear GPU/CPU memory after training

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")  # load Hugging Face token from .env file


# Function to preprocess and batch multimodal data (image + text)
def collate_fn(examples):
  texts = []
  images = []
  for example in examples:
      image = example["image"]
      if image.mode != 'RGB':  # ensure all images are RGB
        image = image.convert('RGB')
      question = example["question"]
      answer = example["multiple_choice_answer"]
      # Build conversation-style input
      messages = [
          {
              "role": "user",
              "content": [
                  {"type": "text", "text": "Answer briefly."},
                  {"type": "image"},
                  {"type": "text", "text": question}
              ]
          },
          {
              "role": "assistant",
              "content": [
                  {"type": "text", "text": answer}
              ]
          }
      ]
      text = processor.apply_chat_template(messages, add_generation_prompt=False)
      texts.append(text.strip())
      images.append([image])

  # Tokenize and pad the text and image inputs
  batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
  labels = batch["input_ids"].clone()
  # Ignore padding and image tokens in loss calculation
  labels[labels == processor.tokenizer.pad_token_id] = -100
  labels[labels == image_token_id] = -100
  batch["labels"] = labels

  return batch


if __name__ == "__main__":

    # Optional quantization config (for 4-bit training to save GPU memory)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print("Load model")
    # Load model with optional quantization and flash attention
    model = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM-256M-Instruct",
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,  # optional: remove if training full precision
        # _attn_implementation="flash_attention_2",  # faster attention implementation
        device_map="auto"  # automatically place layers on GPU(s)
    )

    print(f"device: {model.device}")
    # Load processor (handles both text and image inputs)
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct", token=HF_TOKEN)

    # Get <image> token id to mask it during loss computation
    image_token_id = processor.tokenizer.additional_special_tokens_ids[
                processor.tokenizer.additional_special_tokens.index("<image>")]

    # LoRA configuration for parameter-efficient fine-tuning
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
        use_dora=False,  # disable DoRA variant
        init_lora_weights="gaussian"  # initialize LoRA weights randomly
    )

    lora_config.inference_mode = False  # training mode
    model.add_adapter(lora_config)  # attach LoRA adapter
    model.enable_adapters()
    model = prepare_model_for_kbit_training(model)  # prepare model for 4-bit/8-bit training
    model = get_peft_model(model, lora_config)  # wrap model for PEFT
    print(model.get_nb_trainable_parameters())  # show number of LoRA trainable params

    print("Load datasets")
    # Load small Visual Question Answering dataset from Hugging Face
    dataset = load_dataset('merve/vqav2-small', token=HF_TOKEN, trust_remote_code=True)
    # Split validation data into 50% train / 50% eval
    split_dataset = dataset["validation"].train_test_split(test_size=0.5)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print("Training model")
    # Define training hyperparameters
    training_args = TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # simulate larger batch size
        warmup_steps=50,
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_steps=25,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=1,  # keep only last checkpoint
        optim="paged_adamw_8bit",  # recommended with quantized model; else use adamw_hf
        bf16=True,  # enable bfloat16 precision
        output_dir=f"./model/SmolVLM-256M-Instruct-LORA",  # output directory for checkpoints
        hub_model_id=None,  # optional if pushing to Hugging Face Hub
        report_to="tensorboard",  # enable TensorBoard logging
        remove_unused_columns=False,
        gradient_checkpointing=True,  # reduce memory usage
        label_names=["labels"],
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # Initialize Trainer to handle training & evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,  # custom collate function for multimodal inputs
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Start training
    trainer.train()
    # Save final model and adapters
    trainer.save_model(training_args.output_dir)
    # Clear memory at the end (custom function)
    clear_memory()
