import argparse

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from datallm_engine.engine import BaseEngine
from datallm_engine.config import MODEL_DIR, MODEL_NAME
from datallm_engine.types import EngineResponse

TIMEOUT_KEEP_ALIVE = 60

app = FastAPI()
engine = None


class EngineRequest(BaseModel):
    id: str
    prompts: list[str]
    max_tokens: int | None = Field(default=64, ge=0, le=2048)
    regex: str | None
    temperature: float | None = Field(default=0.7, ge=0, le=2)
    top_p: float | None = Field(default=1, gt=0, le=1)
    cut_window_after: str | None = None


@app.post("/batch-completion", response_model=EngineResponse)
async def batch_completion(engine_request: EngineRequest):
    engine_response = await engine.batch_completion(**engine_request.model_dump())
    return engine_response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()

    engine = BaseEngine(
        MODEL_DIR, MODEL_NAME, tensor_parallel_size=args.tensor_parallel_size
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )
