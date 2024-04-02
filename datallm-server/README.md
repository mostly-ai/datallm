# DataLLM Server
## Run locally
Install dependencies
```shell
poetry install
```

If engine is hosted as a [Modal service](https://modal.com/), set up `modal` (`modal token new`). If engine is hosted on API Server set `ENGINE_API_SERVER_URL` in `main.py`. Then run
```shell
poetry run python -m datallm_server.main --reload
```


## Run from a Docker container
Build the image
```shell
docker build --tag datallm-server .
```

Start the container
```shell
docker run -d -p 80:80 datallm-server
```

If the engine is hosted as a Modal service, then
```shell
docker run -d -e MODAL_TOKEN_ID=$MODAL_TOKEN_ID -e MODAL_TOKEN_SECRET=$MODAL_TOKEN_SECRET -p 80:80 datallm-server
```


## Send API requests
To test a server, set the environment variable `DATALLM_URL` to the server URL:
```shell
export DATALLM_URL=YOUR_DATALLM_URL
```
Then run the following command to send a request:
```shell
curl "http://$DATALLM_URL/row-completions" \
-H 'Content-Type: application/json' \
-d '{
    "model": "mostlyai/datallm-v2-mistral-7b-v0.1",
    "prompt": "Generate age in years",
    "rows": [[{
      "name": "income",
      "value": "5000"
    }]],
    "dtype": "integer",
    "temperature": 0.7
}'
```
