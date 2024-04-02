from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class NamedValue(BaseModel):
    name: str
    value: str


class DtypeEnum(str, Enum):
    string = "string"
    category = "category"
    integer = "integer"
    float = "float"
    boolean = "boolean"
    date = "date"
    datetime = "datetime"


class RowCompletionsRequest(BaseModel):
    rows: List[List[NamedValue]] = Field(
        None,
        description="The existing values used as context for the newly generated values. The returned values will be of same length and in the same order as the provided list of values. Max 100 rows per request are allowed.",
    )
    prompt: str = Field(
        ..., description="The prompt for generating the returned values."
    )
    model: str = Field(
        ...,
        description="The model used for generating new values. Check available models with the models.list endpoint. The default model is the first model in that list.",
    )

    data_description: Optional[str] = Field(
        None,
        description="Additional information regarding the context of the provided values.",
    )
    dtype: DtypeEnum = Field(
        DtypeEnum.string,
        description="The dtype of the returned values. One of `string`, `category`, `integer`, `float`, `boolean`, `date` or `datetime`.",
    )
    regex: Optional[str] = Field(
        None, description="A regex used to limit the generated values."
    )
    categories: Optional[List[str]] = Field(
        None,
        description="The allowed values to be sampled from. If provided, then the dtype is set to `category`.",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=0,
        le=64,
        description="The maximum number of tokens to generate. Only applicable for string dtype.",
    )
    temperature: Optional[float] = Field(
        default=0.7, ge=0.0, le=2.0, description="The temperature used for sampling."
    )
    top_p: Optional[float] = Field(
        default=1.0, gt=0.0, le=1.0, description="The top_p used for nucleus sampling."
    )

    @model_validator(mode="after")
    def validate_parameters(self) -> "RowCompletionsRequest":
        if self.dtype != DtypeEnum.string:
            if self.max_tokens is not None:
                raise ValueError(
                    "`max_tokens` can only be set if using dtype `string`."
                )
        if self.dtype == DtypeEnum.category and not self.categories:
            raise ValueError("Categories must be provided when dtype is category")
        return self


class CompletionUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class RowCompletionsResponse(BaseModel):
    values: List[Optional[str]]
    usage: CompletionUsage
