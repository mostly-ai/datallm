from pydantic import BaseModel


class EngineCompletion(BaseModel):
    text: str
    prompt: str
    prompt_token_ids: list[int]
    token_ids: list[int]
    model: str
    finish_reason: str | None = None


class EngineResponse(BaseModel):
    id: str | None
    engine_completions: list[EngineCompletion]
    completion_time: float | None = None
