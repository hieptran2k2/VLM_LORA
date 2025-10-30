docker run --gpus '"device=0"' \
    --rm \
    -p 8000:8000 \
    --name vllm_vlm_lora_host \
    -e VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 \
    -e TORCH_CUDA_ARCH_LIST=8.0 \
    -v $(pwd):/my_workspace \
    vllm/vllm-openai:v0.10.1 \
    --model /my_workspace/model/Merged_lora \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.50 \
    --max-num-batched-tokens 2048 \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port 8000 \
    --async-scheduling \
    --max-model-len 4094 \