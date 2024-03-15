import pytest

from lmwrapper.structs import LmPrompt

VLLM_UNAVAILABLE = True
try:
    from vllm import LLM, SamplingParams

    from lmwrapper.vllm.wrapper import get_vllm_lm

    VLLM_UNAVAILABLE = False
except ImportError as e:
    print(e)


@pytest.mark.skipif(VLLM_UNAVAILABLE, reason="vLLM not available")
def test_distilgpt2_vllm():
    prompt = LmPrompt(
        "print('Hello world",
        max_tokens=15,
        cache=False,
        temperature=0,
        logprobs=1,
    )
    lm = get_vllm_lm("distilgpt2", tensor_parallel_size=4)
    out = lm.predict(prompt)
    assert out.completion_text


CODE_LLAMAS = [
    "codellama/CodeLlama-7b-hf",
    "codellama/CodeLlama-7b-Instruct-hf",
    "codellama/CodeLlama-13b-hf",
    "codellama/CodeLlama-13b-Instruct-hf",
]

@pytest.mark.skipif(VLLM_UNAVAILABLE, reason="vLLM not available")
@pytest.mark.parametrize("lm", CODE_LLAMAS)
def test_big_models_vllm(lm):
    prompt = LmPrompt(
        "print('Hello world",
        max_tokens=15,
        cache=False,
        temperature=0,
        logprobs=1,
    )
    lm = get_vllm_lm(lm, tensor_parallel_size=4)
    out = lm.predict(prompt)
    assert out.completion_text
