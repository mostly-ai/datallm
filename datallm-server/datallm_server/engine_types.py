import uuid
import interegular
from pydantic import BaseModel, Field
from datallm_server.regex import convert_completion_dtype_to_regex
from datallm_server.prompt import get_prompt_from_components
from datallm_server.types import RowCompletionsRequest, DtypeEnum


class EngineRequest(BaseModel):
    id: str | None
    prompts: list[str]
    regex: str | None
    max_tokens: int = Field(default=64, ge=0, le=2048)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=1, gt=0, le=1)
    cut_window_after: str | None = None

    @classmethod
    def from_row_completion_request(cls, request: RowCompletionsRequest):
        if request.regex:
            regex = request.regex
            try:
                interegular.parse_pattern(regex).to_fsm()
            except interegular.patterns.Unsupported:
                raise ValueError(f"Unsupported regex pattern: {regex}")
            except interegular.patterns.InvalidSyntax:
                raise ValueError(f"Invalid regex pattern: {regex}")
        else:
            regex = convert_completion_dtype_to_regex(
                request.dtype,
                request.categories,
            )

        prompts = [
            get_prompt_from_components(
                prompt=request.prompt,
                data_description=request.data_description,
                dtype=request.dtype,
                values=value,
                categories=request.categories,
            )
            for value in request.rows
        ]

        kwargs = dict(
            id=str(uuid.uuid4().hex),
            prompts=prompts,
            regex=regex,
            cut_window_after="\n### Data Description:\n",
        )
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        return cls(**kwargs)


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
