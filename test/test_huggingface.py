import math

import numpy as np
from lmwrapper.util import StrEnum

import pytest
import torch

from lmwrapper.huggingface_wrapper import Runtime, get_huggingface_lm
from lmwrapper.structs import LmPrompt


class Models(StrEnum):
    CodeT5plus_220M = "Salesforce/codet5p-220m"
    CodeT5plus_6B = "Salesforce/codet5p-6b"
    CodeGen2_1B = "Salesforce/codegen2-1B"
    CodeGen2_3_7B = "Salesforce/codegen2-3_7B"
    InstructCodeT5plus_16B = "Salesforce/instructcodet5p-16b"

    DistilGPT2 = "distilgpt2"
    GPT2 = "gpt2"


CUDA_UNAVAILABLE = not torch.cuda.is_available()
SMALL_GPU = CUDA_UNAVAILABLE or torch.cuda.mem_get_info()[0] < 17_179_869_184  # 16GB

SEQ2SEQ_MODELS = {Models.CodeT5plus_220M}
CAUSAL_MODELS = {Models.DistilGPT2, Models.GPT2, Models.CodeGen2_1B}
BIG_SEQ2SEQ_MODELS = {Models.CodeT5plus_6B, Models.InstructCodeT5plus_16B}
BIG_CAUSAL_MODELS = {Models.CodeGen2_3_7B}
BIG_MODELS = BIG_SEQ2SEQ_MODELS | BIG_CAUSAL_MODELS
ALL_MODELS = SEQ2SEQ_MODELS | CAUSAL_MODELS | BIG_MODELS


def test_logprobs_codegen2():
    lm = get_huggingface_lm(Models.CodeGen2_1B, runtime=Runtime.PYTORCH)
    prompt = LmPrompt(
        "def hello_world():\n   print('",
        max_tokens=15,
        cache=False,
        temperature=0,
    )
    outa = lm.predict(prompt)
    np_logprobsa = np.array(outa.completion_logprobs)
    expa = np.exp(np_logprobsa)
    meana = expa.mean()

    prompt = LmPrompt(
        "def hello_world():\n   print('",
        max_tokens=15,
        cache=False,
        temperature=0,
        patch_model_forward=True
    )
    outb = lm.predict(prompt)
    np_logprobsb = np.array(outb.completion_logprobs)
    expb = np.exp(np_logprobsb)
    meanb = expb.mean()

    assert np.allclose(np_logprobsa, np_logprobsb, atol=0.001, rtol=0.001)



def test_stop_token_removal():
    prompt_str = """Please list the capitals of the following countries

1. Germany
2. USA
3. France
4. Mexico"""
    # Load model
    lm = get_huggingface_lm(Models.DistilGPT2, runtime=Runtime.PYTORCH)

    # Let's make sure we get a stop token in our prompt normally
    prompt = LmPrompt(
        prompt_str,
        max_tokens=15,
        cache=False,
        temperature=0,
    )
    out = lm.predict(prompt)
    assert "Italy" in out.completion_text

    prompt = LmPrompt(
        prompt_str, max_tokens=15, cache=False, temperature=0, stop=["Italy"]
    )
    out = lm.predict(prompt)
    assert "Italy" not in out.completion_text

    prompt_str = """Repeat the following document \"Bob said 'I like to eat candy' and then dove into the pile of candy\""""

    # Let's make sure we get a stop token in our prompt normally
    prompt = LmPrompt(
        prompt_str,
        max_tokens=300,
        cache=False,
        temperature=0,
    )
    out = lm.predict(prompt)
    assert "I like to eat candy" in out.completion_text

    # Let's make sure we get a stop token in our prompt normally
    prompt = LmPrompt(
        prompt_str,
        max_tokens=300,
        cache=False,
        temperature=0,
        stop=["I like to eat candy"],
    )
    out = lm.predict(prompt)
    assert "I like to eat candy" not in out.completion_text


def test_stop_tokens():
    # Load model
    lm = get_huggingface_lm(Models.DistilGPT2, runtime=Runtime.PYTORCH)

    # Let's make sure we get a stop token in our prompt normally
    prompt = LmPrompt(
        "place a newline here",
        max_tokens=15,
        cache=False,
        temperature=0,
    )
    out = lm.predict(prompt)
    assert "\n\n" in out.completion_text

    # Now let's try with one character
    prompt = LmPrompt(
        "place a newline here",
        max_tokens=15,
        cache=False,
        temperature=0,
        stop=["\n"],
    )
    out = lm.predict(prompt)
    assert "\n\n" not in out.completion_text

    # Now with two
    prompt = LmPrompt(
        "place a newline here",
        max_tokens=15,
        cache=False,
        temperature=0,
        stop=["\n\n"],
    )
    out = lm.predict(prompt)
    assert "\n\n\n" not in out.completion_text

    # Now let's try with a sequence longer than the input
    prompt = LmPrompt(
        "place a newline here",
        max_tokens=15,
        cache=False,
        temperature=0,
        stop=["\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"],
    )
    out = lm.predict(prompt)
    assert "\n\n\n" in out.completion_text

    # Now let's try multiple
    prompt = LmPrompt(
        "place a newline here",
        max_tokens=15,
        cache=False,
        temperature=0,
        stop=["\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n", "blah", "\n"],
    )
    out = lm.predict(prompt)
    assert "\n\n\n" not in out.completion_text


def test_distilgpt2_pytorch_runtime():
    prompt = LmPrompt(
        "print('Hello world",
        max_tokens=15,
        cache=False,
        temperature=0,
    )
    lm = get_huggingface_lm(Models.DistilGPT2, runtime=Runtime.PYTORCH)
    out = lm.predict(prompt)
    assert out.completion_text


@pytest.mark.slow()
@pytest.mark.parametrize("lm", ALL_MODELS)
def test_all_pytorch_runtime(lm: str):
    if SMALL_GPU and lm in BIG_MODELS:
        pytest.skip(
            f"Skipped model '{lm}' as model too large for available GPU memory."
        )
    prompt = LmPrompt(
        "print('Hello world",
        max_tokens=15,
        cache=False,
        temperature=0,
    )
    lm = get_huggingface_lm(lm, runtime=Runtime.PYTORCH)
    out = lm.predict(prompt)
    assert out.completion_text


@pytest.mark.slow()
@pytest.mark.skip()  # ORT is not ready yet
@pytest.mark.parametrize("runtime", [Runtime.ORT_CPU, Runtime.ORT_CUDA])
@pytest.mark.parametrize("lm", ALL_MODELS)
def test_get_ort(runtime: Runtime, lm: str):
    prompt = LmPrompt(
        "print('Hello world",
        max_tokens=1,
        cache=False,
        temperature=0,
    )
    lm = get_huggingface_lm(lm, runtime=runtime)
    out = lm.predict(prompt)
    assert out.completion_text


@pytest.mark.slow()
@pytest.mark.skip()  # Better Transformer is not ready yet
@pytest.mark.parametrize("lm", [Models.DistilGPT2, Models.GPT2])
def test_get_better_transformer(lm):
    prompt = LmPrompt(
        "print('Hello world",
        max_tokens=1,
        cache=False,
        temperature=0,
    )
    lm = get_huggingface_lm(lm, runtime=Runtime.BETTER_TRANSFORMER)
    out = lm.predict(prompt)
    assert out.completion_text


@pytest.mark.slow()
@pytest.mark.skip()  # Better Transformer is not ready yet
def test_codegen2_predict_bt():
    lm = Models.CodeGen2_1B
    with pytest.raises(Exception) as e_info:
        get_huggingface_lm(lm, runtime=Runtime.BETTER_TRANSFORMER)
        assert str(e_info.value).startswith("WARNING BetterTransformer")


@pytest.mark.slow()
@pytest.mark.skip()  # TensorRT is not ready yet
@pytest.mark.parametrize("lm", CAUSAL_MODELS)
@pytest.mark.skipif(
    CUDA_UNAVAILABLE,
    reason="Cannot test ORT/ONNX CUDA runtime without CUDA",
)
def test_get_tensorrt(lm: str):
    get_huggingface_lm(lm, runtime=Runtime.ORT_TENSORRT)
