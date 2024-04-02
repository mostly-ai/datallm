from pathlib import Path

from modal import Image, Secret
from datallm_engine.config import MODEL_DIR, MODEL_NAME


def download_model_from_hf(model_name, model_dir):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    snapshot_download(
        model_name,
        local_dir=model_dir,
        ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
    )
    move_cache()


def download_model():
    download_model_from_hf(MODEL_NAME, MODEL_DIR)


vllm_image = (
    Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .pip_install(
        "vllm==0.3.3",
        "huggingface_hub==0.19.4",
        "hf-transfer==0.1.4",
        "lm-format-enforcer==0.8.3",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model,
        secrets=[Secret.from_name("my-huggingface-secret")],
        timeout=60 * 30,
    )
)
