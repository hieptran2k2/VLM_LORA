# VLM_LORA

Implementation of a **Vision-Language Model (VLM)** fine-tuning pipeline using **LoRA**, followed by **efficient inference hosting** via **vLLM** or **llama.cpp**.

---

# 📑 Table of Contents

1. [Usage](#usage)
2. [Fine-tune with LoRA](#finetune-with-lora)
3. [Host with vLLM](#host-with-vllm)
4. [Host with llama.cpp](#host-with-llamacpp)
5. [Testing](#test)

---

## 🚀 Usage

This repository provides both an **implementation** and a **practical guide** for fine-tuning the **SmolVLM Vision-Language Model (VLM)** using **LoRA**, and deploying it efficiently for inference using **vLLM** or **llama.cpp**.

**SmolVLM** is a lightweight Vision-Language Model that supports multimodal reasoning tasks such as:

* 🖼️ Visual Question Answering (VQA)
* 📝 Image Captioning
* 📖 Visual Storytelling and Reasoning

This project demonstrates how to fine-tune SmolVLM on a **small VQA dataset** using **LoRA adapters**, and how to **save and deploy** the trained model efficiently.

For detailed fine-tuning instructions, see [Fine-tune with LoRA](#finetune-with-lora).

---

## 🧠 Fine-tune with LoRA

### Environment Setup

* **Python 3.10+**

* **Environment variables:**
  Create a `.env` file in your project root directory:

  ```bash
  HF_TOKEN=your_huggingface_access_token
  ```

* **Library Installation**

  * Install **PyTorch with CUDA support**:

    ```bash
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
    ```

  * Install **Flash Attention**:

    ```bash
    pip install flash-attn --no-build-isolation
    ```

  * Install other dependencies:

    ```bash
    pip install -r requirements.txt
    ```

---

### Fine-tuning Configuration and Components

<details>
<summary>⚙️ Expand Configuration Details</summary>

---

### 1️⃣ BitsAndBytes Quantization Config (`bnb_config`)

Used to **reduce GPU memory consumption** by quantizing model weights to **4-bit precision**.

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

💡 **Tip:**
If you have sufficient GPU VRAM, you can disable quantization by commenting out this config and removing `quantization_config=bnb_config` when loading the model.

---

### 2️⃣ Flash Attention (Optional)

For **faster training and inference**, Flash Attention v2 can be used.

```python
# _attn_implementation="flash_attention_2"
```

To enable it, uncomment the line above when loading your model.
If your GPU does **not** support Flash Attention, keep it commented out.

---

### 3️⃣ LoRA Configuration

Defines which layers are fine-tuned and how LoRA adapters are applied:

```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=[
        'down_proj', 'o_proj', 'k_proj',
        'q_proj', 'gate_proj', 'up_proj', 'v_proj'
    ],
    use_dora=False,
    init_lora_weights="gaussian"
)
```

💡 **Notes:**

* Increase `r` and `lora_alpha` for higher adapter capacity (improves accuracy but increases memory usage).
* Adjust `target_modules` to match your model architecture.
* Set `use_dora=True` to enable **DoRA**, an experimental variant of LoRA.

---

### 4️⃣ Dataset & Data Collation

This example uses [`merve/vqav2-small`](https://huggingface.co/datasets/merve/vqav2-small), a compact Visual Question Answering dataset.

A custom **collate function** prepares image-text pairs and applies the chat template:

```python
def collate_fn(examples):
    texts, images = [], []
    for ex in examples:
        image = ex["image"].convert("RGB")
        question = ex["question"]
        answer = ex["multiple_choice_answer"]
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Answer briefly."},
                {"type": "image"},
                {"type": "text", "text": question}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": answer}
            ]}
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=False)
        texts.append(text.strip())
        images.append([image])

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    batch["labels"] = labels
    return batch
```

🧩 **If using another dataset:**
Modify the dataset field names (`question`, `answer`, etc.) and adjust preprocessing accordingly.

---

### 5️⃣ Training Arguments

Defines the main hyperparameters for training:

```python
training_args = TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 16
    warmup_steps=50,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=25,
    save_strategy="steps",
    save_steps=250,
    save_total_limit=1,
    optim="paged_adamw_8bit",
    bf16=True,
    output_dir="./model/SmolVLM-256M-Instruct-LORA",
    report_to="tensorboard",
    remove_unused_columns=False,
    gradient_checkpointing=True,
    label_names=["labels"],
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
)
```

🔧 **Adjustable parameters:**

* `num_train_epochs` → number of epochs
* `per_device_train_batch_size` → per-GPU batch size
* `learning_rate` → typical range: 1e-4 to 2e-5
* `output_dir` → checkpoint storage path

---

</details>

---

## 🏃 Run Training

### Step 1: Prepare `.env`

```bash
HF_TOKEN=your_huggingface_access_token
```

### Step 2: Start training

```bash
python train.py
```

### Step 3: Monitor training

TensorBoard logs are available in:

```bash
tensorboard --logdir SmolVLM-256M-Instruct-LORA/runs/ --host=localhost --port=1234
```

---

## 💾 Output

After training completes, LoRA adapter checkpoints will be saved in:

```
./model/SmolVLM-256M-Instruct-LORA/
```

To run inference:

```bash
python infer.py --lora_path /path/to/LoRA \
                --prompt "Your prompt" \
                --image_path /path/to/input_image
```

---

## 🧩 Host with vLLM

<details>
<summary>Expand vLLM Deployment Guide</summary>

vLLM **does not automatically load LoRA adapters** for **vision encoder** layers.
To ensure all LoRA adapters are applied correctly, **merge LoRA** into the model **after training**.

### 🔧 Merge LoRA into Model

```bash
python export/merge_lora.py --lora_path /path/to/LoRA --output_path /path/to/save/model
```

### 🚀 Host the Model via vLLM

```bash
docker run --gpus '"device=0"' \
    --rm \
    -p 8000:8000 \
    --name vllm_vlm_lora_host \
    -e VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 \
    -e TORCH_CUDA_ARCH_LIST=8.0 \
    -v $(pwd):/my_workspace \
    vllm/vllm-openai:v0.10.1 \
    --model /my_workspace/{output_save_model_merge_lora} \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.50 \
    --max-num-batched-tokens 2048 \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port 8000 \
    --async-scheduling \
    --max-model-len 4094
```

</details>

---

## 🧠 Host with llama.cpp

<details>
<summary>Expand llama.cpp Deployment Guide</summary>

Since **llama.cpp** requires **separated LLM and vision encoder blocks**, merging LoRA beforehand ensures complete adapter integration.

### 🔧 Merge LoRA into Model

```bash
python export/merge_lora.py --lora_path /path/to/LoRA --output_path /path/to/save/model
```

### 🧱 Export Model to GGUF

Run Docker container:

```bash
docker run -it --gpus '"device=0"' \
    --rm \
    -p 8080:8080 \
    --name llama_cpp_export_gguf \
    -v $(pwd):/my_workspace \
    --entrypoint bash \
    ghcr.io/ggml-org/llama.cpp:full-cuda
```

Convert to GGUF format:

```bash
python3 convert_hf_to_gguf.py /my_workspace/output_merged_lora --outfile /my_workspace/output_merged_lora_gguf/model.gguf --outtype bf16
python3 convert_hf_to_gguf.py /my_workspace/output_merged_lora --mmproj --outfile /my_workspace/output_merged_lora_mmproj_gguf/model.gguf --outtype bf16
```

(Optional) Quantize:

```bash
quantize /my_workspace/output_merged_lora_gguf/model.gguf /my_workspace/output_merged_lora_gguf-q4_K_M/model-q4_K_M.gguf q4_K_M
```

### 🚀 Host Model in llama.cpp

```bash
docker run -it --gpus '"device=0"' \
    --rm \
    -p 8080:8080 \
    --name llama_cpp_vlm_lora_host \
    -v $(pwd):/my_workspace \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server \
    -m /my_workspace/output_merged_lora_gguf/model.gguf \
    --mmproj /my_workspace/output_merged_lora_mmproj_gguf/model.gguf \
    -np 4 \
    --host 0.0.0.0 \
    --port 8080 \
    --n-gpu-layers -1 \
    --ctx-size 2048 \
    --flash-attn on
```

</details>

---

## 🧪 Test

<details>
<summary>Expand Testing Instructions</summary>

### ✅ Inference via vLLM Hosting

```bash
python test/test_vllm.py --url http://url_for_vllm_host \
                         --model /my_workspace/{output_save_model_merge_lora} \
                         --image_path /path/to/image \
                         --prompt "Your prompt"
```

### ✅ Inference via llama.cpp Hosting

```bash
python test/test_llama.py --url http://url_for_llama_host \
                          --image_path /path/to/image \
                          --prompt "Your prompt"
```

### ⚡ Benchmark Throughput (Multi-request)

```bash
python test/multi-request.py --url http://url_for_llama_or_vllm_host \
                             --root_dir /path/to/image_directory \
                             --host "vllm"  # or "llama_cpp" \
                             --model /my_workspace/{output_save_model_merge_lora} \
                             --prompt "Describe the image in detail." \
                             --num_requests 100 \
                             --concurrent 20
```

</details>

---

## 🔗 Reference

**Base Model:** [HuggingFaceTB/SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)
**vLLM:** [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
**llama.cpp:** [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
