from datallm_server.engine_types import EngineRequest, EngineResponse
from functools import lru_cache
from fastapi import HTTPException
import modal


@lru_cache(maxsize=10)
def get_modal_func(app_name: str, tag: str) -> modal.Function:
    # Takes ~140ms to lookup, use cache to remove latency
    try:
        return modal.Function.lookup(app_name, tag)
    except modal.exception.NotFoundError:
        raise RuntimeError("Error looking up modal function.")


async def call_modal_engine_fn(
    modal_func: modal.Function, request: EngineRequest
) -> EngineResponse:
    request_dict = request.model_dump()
    try:
        response = await modal_func.remote.aio(**request_dict)
        validated_engine_response = EngineResponse.model_validate_json(
            response
        )  # ignore type hint for response
        return validated_engine_response
    except Exception as _:
        raise RuntimeError("Error calling the engine.")


async def call_engine(request: EngineRequest, model: str) -> EngineResponse:
    modal_app_name = f"datallm-engine-{model.replace('/', '--')}"
    tag = "DataLLMEngine.batch_completion"

    def clear_and_retry():
        get_modal_func.cache_clear()
        return get_modal_func(modal_app_name, tag)

    try:
        func = get_modal_func(modal_app_name, tag)
        try:
            return await call_modal_engine_fn(func, request)
        except RuntimeError as e:
            # Attempt to clear the cache and retry once if there's an error calling the engine
            func = clear_and_retry()
            return await call_modal_engine_fn(func, request)
    except RuntimeError:
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again later.",
        )
