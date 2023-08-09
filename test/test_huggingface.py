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
def test_get_better_transformer_codegen(lm):
    get_huggingface_lm(lm, runtime=Runtime.PYTORCH)
    get_huggingface_lm(lm, runtime=Runtime.BETTER_TRANSFORMER)


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
