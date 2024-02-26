import pytest

from lmwrapper.structs import LmPrompt

VLLM_UNAVAILABLE = True
try:
    from vllm import LLM, SamplingParams

    from lmwrapper.vllm.wrapper import get_vllm_lm

    VLLM_UNAVAILABLE = False
except ImportError:
    pass


@pytest.mark.skipif(VLLM_UNAVAILABLE, reason="vLLM not available")
def test_distilgpt2_vllm():
    prompt = LmPrompt(
        "print('Hello world",
        max_tokens=15,
        cache=False,
        temperature=0,
    )
    lm = get_vllm_lm("distilgpt2")
    out = lm.predict(prompt)
    assert out.completion_text
