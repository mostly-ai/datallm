from typing import Any, List, Optional, Union

from datallm._base_client import POST, SyncClient, GET
from datallm._types import (
    NamedValue,
    DtypeEnum,
    RowCompletionsRequest,
    RowCompletionsResponse,
)


class SyncRowCompletionsResource:
    _client: SyncClient
    _path: Union[str, List[Any]]

    def __init__(self, client: SyncClient):
        self._client = client
        self._path = "row-completions"

    def create(
        self,
        model: str,
        prompt: str,
        rows: Optional[List[List[NamedValue]]],
        data_description: Optional[str],
        dtype: DtypeEnum,
        regex: Optional[str],
        categories: Optional[List[str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> RowCompletionsResponse:
        request = RowCompletionsRequest(
            model=model,
            prompt=prompt,
            dtype=dtype,
            data_description=data_description,
            rows=rows,
            regex=regex,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            categories=categories,
        )
        return self._client.send_request(
            path=self._path,
            json=request.model_dump(),
            verb=POST,
            response_type=RowCompletionsResponse,
        )


class SyncModelsResource:
    _client: SyncClient
    _path: Union[str, List[Any]]

    def __init__(self, client: SyncClient):
        self._client = client
        self._path = "models"

    def list(self) -> RowCompletionsResponse:
        return self._client.send_request(
            path=self._path,
            verb=GET,
            response_type=List[str],
        )
