FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS datallm-engine-base

RUN apt-get update -y && apt-get install -y python3-pip \
    && pip3 install --no-cache-dir vllm==0.3.3 lm-format-enforcer==0.8.3 huggingface_hub[cli] \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN mkdir /model \
    && huggingface-cli download 'mostlyai/datallm-v2-mistral-7b-v0.1' --local-dir /model
COPY . /datallm-engine
WORKDIR /datallm-engine
RUN pip3 install --no-cache-dir -e .
ENV DATALLM_ENGINE_API_SERVER=true
ENTRYPOINT ["python3", "-m", "datallm_engine.entrypoints.api_server.serve"]
