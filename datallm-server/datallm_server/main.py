import logging
import os
import time
import traceback
from typing import List

from urllib.request import Request

import uvicorn
import fastapi
from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from datallm_server.coerce import coerce_dtype
from datallm_server.engine_types import EngineRequest, EngineResponse

from datallm_server.types import (
    CompletionUsage,
    RowCompletionsRequest,
    RowCompletionsResponse,
)

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)-7s: %(message)s"
)
app = FastAPI()

INTERNAL_PATHS = ["/health"]
AVAILABLE_MODELS = [
    "mostlyai/datallm-v2-mistral-7b-v0.1",
    "mostlyai/datallm-v2-mixtral-8x7b-v0.1",
    "mostlyai/datallm-v2-meta-llama-3-8b",
]
MAX_BATCH_SIZE = 100
ENGINE_API_SERVER_URL = None  # If None assume Modal engine is used

if ENGINE_API_SERVER_URL is None:
    from datallm_server.modal_engine import call_engine
else:
    import httpx


    async def call_engine(engine_request: EngineRequest, model: str) -> EngineResponse:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{ENGINE_API_SERVER_URL}/batch-completion",
                json=engine_request.model_dump(),
            )
            r.raise_for_status()
        return EngineResponse(**r.json())


@app.middleware("http")
async def check_api_gateway_secret(request: Request, call_next):
    if request.url.path in INTERNAL_PATHS:
        response = await call_next(request)
        return response
    api_gateway_local_secret = os.environ.get("DATALLM_API_GATEWAY_SECRET", None)
    api_gateway_secret = request.headers.get("api-gateway-secret", None)
    if (
            api_gateway_local_secret is not None
            and api_gateway_local_secret != api_gateway_secret
    ):
        return JSONResponse(
            status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
            content={"error": "Unauthorized, invalid " "`api-gateway-secret` header."},
        )
    response = await call_next(request)
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logging.error(traceback.format_exc())
        raise e


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.get("/health", include_in_schema=False)
async def health() -> Response:
    return Response(status_code=200)


@app.get("/models", response_model=list[str])
def list_models() -> List[str]:
    return AVAILABLE_MODELS


def row_completion_response_from_engine_response(
        engine_response: EngineResponse, completion_request: RowCompletionsRequest
):
    values = [
        engine_completion.text.strip()
        for engine_completion in engine_response.engine_completions
    ]
    values = coerce_dtype(values, completion_request.dtype)
    usage = completion_usage_from_engine_response(engine_response)
    return RowCompletionsResponse(
        values=values,
        usage=usage,
    )


def completion_usage_from_engine_response(
        engine_response: EngineResponse,
) -> CompletionUsage:
    num_prompt_tokens = 0
    num_completion_tokens = 0
    for engine_completion in engine_response.engine_completions:
        num_prompt_tokens += len(engine_completion.prompt_token_ids)
        num_completion_tokens += len(engine_completion.token_ids)
    return CompletionUsage(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_completion_tokens,
        total_tokens=num_prompt_tokens + num_completion_tokens,
    )


@app.post(
    "/row-completions",
    response_model=RowCompletionsResponse,
    summary="Generate row completions.",
    description="Generate completions for a given prompt and rows.",
    response_description="The generated completions and their token usage.",
)
async def create_row_completions(request: RowCompletionsRequest):
    if request.model not in AVAILABLE_MODELS:
        return JSONResponse(
            status_code=fastapi.status.HTTP_400_BAD_REQUEST,
            content={
                "detail": "Invalid model, must be one of: "
                          + ", ".join(AVAILABLE_MODELS)
            },
        )
    if len(request.rows) > MAX_BATCH_SIZE:
        return JSONResponse(
            status_code=fastapi.status.HTTP_400_BAD_REQUEST,
            content={"detail": f"Too many rows, must be at most {MAX_BATCH_SIZE}"},
        )
    # logging.info(f"row-completions-request: {request}")
    try:
        engine_request = EngineRequest.from_row_completion_request(request)
    except ValueError as e:
        return JSONResponse(
            status_code=fastapi.status.HTTP_400_BAD_REQUEST,
            content={"detail": str(e)},
        )
    #     logging.info(f"engine-request: {engine_request}")
    engine_response = await call_engine(engine_request, request.model)
    #     logging.info(f"engine-response: {engine_response}")
    response = row_completion_response_from_engine_response(engine_response, request)
    #     logging.info(f"row-completions-response: {response}")
    return response


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8000, type=int)

    args = parser.parse_args()
    uvicorn.run(
        app="datallm_server.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        timeout_keep_alive=30,
    )
