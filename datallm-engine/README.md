# DataLLM Engine
The engine for serving a Large Language Model (LLM) for row completion.

## Deploy
There are two ways to deploy the engine, either as a REST API or as a [Modal service](https://modal.com/).

### Alternative 1: Modal

1. Get a `Modal` account.

2. In the Modal console, create a new secret `my-huggingface-secret` and add your Hugging Face token `HF_TOKEN` in it.

3. Install the dependencies with `poetry install --with modal`, activate poetry shell and set up token/secret `modal token new`.

4. To deploy (note `keep_warm` is set to one, so it will provision a GPU which will cost credits per hour)
   ```shell
   modal deploy datallm_engine/entrypoints/modal/serve.py
   ```
   Change `GPU_CONFIG` to alter the GPU used for deployment. If you have several Modal environments (e.g. `dev` and `main`) you will need to specify which environment to deploy to `--environment dev`.

### Alternative 2: REST API

1. Building the Dockerfile requires an Nvidia GPU, toolkit, and container toolkit. To build, run the following command:
   ```shell
   docker build -t datallm-engine .
   ```

2. Run the following command to start the engine server (for reference a server with an A10 GPU can host the default `mostlyai/datallm-v2-mistral-7b-v0.1` model):
   ```shell
   docker run --gpus all -p 8000:8000 datallm-engine
   ```

3. You can then call the engine server using the following CURL command:
   ``` shell
   curl "http://localhost:8000/batch-completion" \
   -H 'Content-Type: application/json' \
   -d '{
       "id": "Some UUID",
       "prompts": [
           "Some prompt",
           "Another prompt"
       ],
       "temperature": 0.7,
       "max_tokens": 10
   }'
   ```
