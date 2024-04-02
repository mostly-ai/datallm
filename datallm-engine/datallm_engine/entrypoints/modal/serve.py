from modal import enter, method, exit, gpu, Stub

from datallm_engine.entrypoints.modal.image import vllm_image, MODEL_DIR, MODEL_NAME
from datallm_engine.engine import BaseEngine

stub = Stub(f"datallm-engine-{MODEL_NAME.replace('/', '--')}", image=vllm_image)
keep_warm = 1
GPU_CONFIG = gpu.H100(
    count=1
)  # change to gpu.A10G(count=1) for lower cost and performance


@stub.cls(
    gpu=GPU_CONFIG,
    timeout=60 * 5,
    concurrency_limit=1,
    container_idle_timeout=60 * 10,
    # allow_concurrent_inputs=2,
    image=vllm_image,
)
class DataLLMEngine:
    @enter()
    def start(self):
        self.model = BaseEngine(
            MODEL_DIR, MODEL_NAME, tensor_parallel_size=GPU_CONFIG.count
        )

    @exit()
    def stop(self):
        if GPU_CONFIG.count > 1:
            import ray

            ray.shutdown()

    @method(keep_warm=keep_warm)
    async def batch_completion(
        self,
        prompts: list[str],
        id=None,
        regex=None,
        max_tokens=64,
        temperature=0.7,
        top_p=1.0,
        cut_window_after=None,
    ) -> str:
        completions = await self.model.batch_completion(
            prompts=prompts,
            id=id,
            regex=regex,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            cut_window_after=cut_window_after,
        )
        return completions.model_dump_json()
