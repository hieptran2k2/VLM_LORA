docker run -it --gpus '"device=0"' \
    --rm \
    -p 2345:8080 \
    --name llama_cpp_vlm_lora_host \
    -v $(pwd):/my_workspace \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server \
    -m /my_workspace/model/Merged_lora_gguf/model.gguf \
    --mmproj /my_workspace/model/Merged_lora_mmproj_gguf/mmproj-model.gguf \
    -np 4 \
    --host 0.0.0.0 \
    --port 8080 \
    --n-gpu-layers -1 \
    --ctx-size 2048 \
    --flash-attn on