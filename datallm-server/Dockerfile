FROM python:3.11 as datallm-server

WORKDIR /code
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="${PATH}:/root/.local/bin"

COPY poetry.lock pyproject.toml /code/
RUN poetry install --no-interaction --no-ansi
COPY datallm_server /code/datallm_server

CMD ["poetry", "run", "uvicorn", "datallm_server.main:app", "--host", "0.0.0.0", "--port", "80", "--timeout-keep-alive", "30"]
