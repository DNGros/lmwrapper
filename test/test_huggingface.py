import pytest

from lmwrapper.huggingface_wrapper import Runtime, get_huggingface_lm
from lmwrapper.structs import LmPrompt
import torch

ALL_MODELS = ["distilgpt2", "gpt2", "Salesforce/codet5p-220m", "Salesforce/codegen2-1B"]

CUDA_UNAVAILABLE = not torch.cuda.is_available()


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_get_pytorch(lm):
    get_huggingface_lm(lm, runtime=Runtime.PYTORCH)


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_get_ort_cpu(lm):
    get_huggingface_lm(lm, runtime=Runtime.ORT_CPU)


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_get_better_transformer(lm):
    get_huggingface_lm(lm, runtime=Runtime.BETTER_TRANSFORMER)


@pytest.mark.parametrize("lm", ["Salesforce/codegen2-1B"])
def test_codegen2_predict(lm):
    prompt = LmPrompt(
        "print('Hello world", max_tokens=1, cache=False, temperature=0
    )
    lm1 = get_huggingface_lm(lm, runtime=Runtime.PYTORCH)
    out1 = lm1.predict(prompt)
    assert out1.completion_text == "!"

@pytest.mark.parametrize("lm", ["Salesforce/codegen2-1B"])
def test_codegen2_predict_bt(lm):
    with pytest.raises(Exception) as e_info:
        get_huggingface_lm(lm, runtime=Runtime.BETTER_TRANSFORMER)
        assert str(e_info.value).startswith("WARNING BetterTransformer")


@pytest.mark.parametrize("lm", ALL_MODELS)
@pytest.mark.skipif(
    CUDA_UNAVAILABLE, reason="Cannot test ORT/ONNX CUDA runtime without CUDA"
)
def test_get_onnx(lm):
    get_huggingface_lm(lm, runtime=Runtime.ONNX)


@pytest.mark.parametrize("lm", ALL_MODELS)
@pytest.mark.skipif(
    CUDA_UNAVAILABLE, reason="Cannot test ORT/ONNX CUDA runtime without CUDA"
)
def test_get_tensorrt(lm):
    get_huggingface_lm(lm, runtime=Runtime.TENSORRT)