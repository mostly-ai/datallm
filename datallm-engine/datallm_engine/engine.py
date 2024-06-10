import time
import asyncio
import os
from pathlib import PosixPath
from typing import Iterable

from datallm_engine.types import EngineCompletion, EngineResponse

import_locally = os.getenv("DATALLM_ENGINE_API_SERVER") in ["True", "true", "TRUE"]

if import_locally:
    from lmformatenforcer.integrations.vllm import (
        build_vllm_token_enforcer_tokenizer_data,
    )
    from vllm.transformers_utils.tokenizer import get_tokenizer
    from vllm import SamplingParams
    from vllm.utils import random_uuid
    from lmformatenforcer.regexparser import RegexParser
    from lmformatenforcer.integrations.vllm import build_vllm_logits_processor
else:
    from datallm_engine.entrypoints.modal.image import vllm_image

    with vllm_image.imports():
        from lmformatenforcer.integrations.vllm import (
            build_vllm_token_enforcer_tokenizer_data,
        )
        from vllm.transformers_utils.tokenizer import get_tokenizer
        from vllm import SamplingParams
        from vllm.utils import random_uuid
        from lmformatenforcer.regexparser import RegexParser
        from lmformatenforcer.integrations.vllm import build_vllm_logits_processor


class BaseEngine:
    def __init__(self, model_dir, model_name: str, tensor_parallel_size: int = 1, environment='main'):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        print("🥶 cold starting inference")
        start = time.monotonic_ns()

        if tensor_parallel_size > 1:
            # Patch issue from https://github.com/vllm-project/vllm/issues/1116
            import ray

            ray.shutdown()
            ray.init(num_gpus=tensor_parallel_size)

        if isinstance(model_dir, PosixPath):
            model_dir = str(model_dir.absolute())
        self.model_dir = model_dir
        self.model_name = model_name
        self.max_model_len = 2048

        engine_args = AsyncEngineArgs(
            model=self.model_dir,
            max_model_len=self.max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.90,
            enforce_eager=False,
            disable_log_requests=False if environment == 'dev' else True,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.template = "{prompt}"

        self.tokenizer = get_tokenizer(self.model_dir)
        self.enforcer_tokenizer_data = build_vllm_token_enforcer_tokenizer_data(
            self.tokenizer
        )

        duration_s = (time.monotonic_ns() - start) / 1e9
        print(f"🏎️ engine started in {duration_s:.0f}s")

    async def batch_completion(
        self,
        prompts: list[str],
        id: str | None = None,
        regex: str | None = None,
        max_tokens: int | None = 64,
        temperature: float = 0.7,
        top_p: float = 1.0,
        cut_window_after: str | None = None,
    ) -> EngineResponse:
        """
        Generate a completion for a given prompt.

        Args:
            prompts: A list of prompts to generate completions for.
            id: The id of the request.
            regex: A regex to enforce on the generated tokens.
            max_tokens: The maximum number of tokens to generate.
            temperature: The temperature to use for sampling.
            top_p: The top_p to use for sampling.
            cut_window_after: A string to cut the prompt after to avoid exceeding the maximum model length.
        """
        logits_processors = []
        if regex:
            parser = RegexParser(regex)
            logits_processors = [
                build_vllm_logits_processor(self.enforcer_tokenizer_data, parser)
            ]

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            logits_processors=logits_processors,
            top_p=top_p,
            stop=["</s>", "\n"],
        )

        t0 = time.time()

        def cut_window_if_too_long(prompt_ids: list[int]):
            if len(prompt_ids) <= self.max_model_len - max_tokens:
                return prompt_ids
            if cut_window_after:
                # NOTE: the [1:] is specific for sentencepiece and may need to be changed for other tokenizers
                start_sequence = self.tokenizer(cut_window_after, add_special_tokens=False).input_ids[1:]
                end_of_start_sequence = None

                for i in range(len(prompt_ids)):
                    if prompt_ids[i:i + len(start_sequence)] == start_sequence:
                        end_of_start_sequence = i + len(start_sequence)
                        break
                if not end_of_start_sequence:
                    raise ValueError(f"`cut_window_after` parameter \"{cut_window_after}\" not found in prompt")
                num_tokens_to_cut = len(prompt_ids) - (self.max_model_len - max_tokens)
                prompt_ids = prompt_ids[:end_of_start_sequence] + prompt_ids[end_of_start_sequence + num_tokens_to_cut:]

            if len(prompt_ids) > self.max_model_len - max_tokens:
                prompt_ids = prompt_ids[-(self.max_model_len - max_tokens):]
            return prompt_ids

        results_generators = []
        for prompt in prompts:
            prompt = self.template.format(prompt=prompt)
            prompt_ids = self.tokenizer(prompt).input_ids
            prompt_ids = cut_window_if_too_long(prompt_ids)
            engine_request_id = random_uuid()
            result_generator = self.engine.generate(
                request_id=engine_request_id,
                prompt=None,
                prompt_token_ids=prompt_ids,
                sampling_params=sampling_params,
            )
            results_generators.append(result_generator)

        async def consume_result_generator(result_generator):
            async for result in result_generator:
                final_request_output = result
            return final_request_output

        final_outputs = await asyncio.gather(
            *[
                consume_result_generator(result_generator)
                for result_generator in results_generators
            ]
        )

        batch_completion_time = time.time() - t0
        responses = []
        for i, final_output in enumerate(final_outputs):
            request_output = final_output.outputs[0]
            response = EngineCompletion(
                text=request_output.text,
                prompt=prompts[i],
                prompt_token_ids=final_output.prompt_token_ids,
                token_ids=request_output.token_ids,
                model=self.model_name,
                finish_reason=request_output.finish_reason,
            )
            responses.append(response)
        return EngineResponse(
            id=id, engine_completions=responses, completion_time=batch_completion_time
        )
