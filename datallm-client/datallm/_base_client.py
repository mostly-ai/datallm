import os
from typing import Any, List, Literal, Optional, Union

import httpx

from datallm._exceptions import APIError, APIStatusError

GET = "get"
POST = "post"
PATCH = "patch"
DELETE = "delete"
HttpVerb = Literal[GET, POST, PATCH, DELETE]  # type: ignore

DEFAULT_BASE_URL = "https://data.mostly.ai"


def parse_response(
    response: httpx.Response,
    response_type: type = dict,
    extra_key_values: Optional[dict] = None,
) -> Any:
    """
    Pack the response into: a specific type (e.g. Pydantic class) with the optional inclusion of:
        - extra_key_values - to store extra information, that is potentially used via that object
    :param response:
    :param response_type: a specific type to return (e.g. Pydantic class)
    :param extra_key_values: Any extra information storage to include in the returned object
    :return: response in a designated type with optional extras
    """
    response_json = response.json()
    if response.content:
        if isinstance(extra_key_values, dict) and isinstance(response_json, dict):
            response_json["extra_key_values"] = extra_key_values
        return (
            response_type(**response_json)
            if isinstance(response_json, dict)
            else response_json
        )
    else:
        return None


class _BaseClient:
    """
    Base client class, which contains all the essentials to be used by sub-classes.
    """

    API_SECTION = []  # "api", "v1"
    SECTION = []

    def __init__(
        self, http_client, base_url: Optional[str] = None, api_key: Optional[str] = None
    ):
        self.base_url = (
            base_url or os.getenv("DATALLM_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.api_key = api_key or os.getenv("DATALLM_API_KEY")
        self.http_client = http_client
        if not self.api_key:
            raise APIError(
                "The API key must be either set by passing api_key to the client or by specifying a "
                "DATALLM_API_KEY environment variable"
            )

    def headers(self):
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def request(
        self,
        path: Union[str, List[Any]],
        verb: HttpVerb = "get",
        **kwargs,
    ) -> httpx.Request:
        """
        This method is rather an extended request helper method, which could be split into two:
        :param path: a single str, or a list of parts of the path to concatenate
        :param verb: get/post/patch/delete
        :param kwargs: httpx's request function's kwargs
        :return: response in a designated type with optional extras
        """

        path_list = [path] if isinstance(path, str) else [str(p) for p in path]
        prefix = self.API_SECTION + self.SECTION
        full_path = [self.base_url] + prefix + path_list
        full_url = "/".join(full_path)

        kwargs["headers"] = kwargs.get("headers") or {}
        kwargs["headers"] |= self.headers()
        request = self.http_client.build_request(verb, full_url, **kwargs)  # type: ignore
        return request

    def json_post_request(self, **kwargs):
        headers = {"Content-Type": "application/json"}
        return self.request(verb=POST, headers=headers, **kwargs)


class SyncClient(_BaseClient):
    def __init__(self, base_url=None, api_key=None):
        timeout = httpx.Timeout(10.0, connect=60.0, read=45.0)
        http_client = httpx.Client(timeout=timeout)
        super().__init__(base_url=base_url, api_key=api_key, http_client=http_client)

    def send_request(
        self,
        path: Union[str, List[Any]],
        verb: HttpVerb,
        response_type: type = dict,
        extra_key_values: Optional[dict] = None,
        **kwargs,
    ) -> Any:
        """
        Create a request with its params and execute it; Raise an exception in case of an unsuccessful result
        :return: response in a designated type with optional extras
        """
        request = self.request(path, verb, **kwargs)
        try:
            response = self.http_client.send(request)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # Handle HTTP errors (not in 2XX range)
            raise APIStatusError(
                f"HTTP error occurred: {exc.response.status_code} {exc.response.content}"
            ) from exc
        except httpx.RequestError as exc:
            # Handle request errors (e.g., network issues)
            raise APIError(
                f"An error occurred while requesting {exc.request.url!r}."
            ) from exc
        response = parse_response(
            response, response_type=response_type, extra_key_values=extra_key_values
        )
        return response
