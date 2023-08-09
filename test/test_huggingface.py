import pytest

from lmwrapper.huggingface_wrapper import Runtime, get_huggingface_lm
from lmwrapper.structs import LmPrompt
import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
from optimum.pipelines import pipeline

CUDA_UNAVAILABLE = not torch.cuda.is_available()
SMALL_GPU = CUDA_UNAVAILABLE or torch.cuda.mem_get_info()[0] < 17179869184  # 16GB

if SMALL_GPU:
    ALL_MODELS = ["distilgpt2", "gpt2", "Salesforce/codet5p-220m", "Salesforce/codegen2-1B"]
    CAUSAL_MODELS = ["distilgpt2", "gpt2", "Salesforce/codegen2-1B"]
else:
    ALL_MODELS = ["distilgpt2", "gpt2", "Salesforce/codet5p-6b", "Salesforce/codegen2-3_7B"]
    CAUSAL_MODELS = ["distilgpt2", "gpt2", "Salesforce/codegen2-3_7B"]

# @pytest.mark.skipif(
#     CUDA_UNAVAILABLE, reason="Cannot test ORT/ONNX CUDA runtime without CUDA"
# )
# def test_onnx_works():
#     ort_model = ORTModelForSequenceClassification.from_pretrained(
#       "distilbert-base-uncased-finetuned-sst-2-english",
#       export=True,
#       provider="CUDAExecutionProvider",
#     )

#     tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
#     inputs = tokenizer("Both the music and visual were astounding, not to mention the actors performance.", return_tensors="pt", padding=False)
#     outputs = ort_model(**inputs)
#     assert outputs
#     assert ort_model.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]

# def test_onnx_works_cpu():
#     ort_model = ORTModelForSequenceClassification.from_pretrained(
#       "distilbert-base-uncased-finetuned-sst-2-english",
#       export=True,
#       provider="CPUExecutionProvider",
#     )

#     tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
#     inputs = tokenizer("Both the music and visual were astounding, not to mention the actors performance.", return_tensors="pt", padding=False)
#     outputs = ort_model(**inputs)
#     assert outputs
#     assert ort_model.providers == ["CPUExecutionProvider"]

@pytest.mark.parametrize("lm", ALL_MODELS)
def test_get_pytorch(lm):
    get_huggingface_lm(lm, runtime=Runtime.PYTORCH)


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_get_ort_cpu(lm):
    get_huggingface_lm(lm, runtime=Runtime.ORT_CPU)

@pytest.mark.parametrize("lm", ALL_MODELS)
def test_get_ort_gpu(lm):
    prompt = LmPrompt("print('Hello world", max_tokens=1, cache=False, temperature=0)
    lm1 = get_huggingface_lm(lm, runtime=Runtime.ONNX)
    out1 = lm1.predict(prompt)
    assert out1.completion_text == "!'"


@pytest.mark.parametrize("lm", ["distilgpt2", "gpt2"])
def test_get_better_transformer(lm):
    get_huggingface_lm(lm, runtime=Runtime.BETTER_TRANSFORMER)

@pytest.mark.parametrize("runtime", [Runtime.PYTORCH, Runtime.BETTER_TRANSFORMER ])
def test_gpt2_predict(runtime):
    lm = "gpt2"
    prompt = LmPrompt("print('Hello world", max_tokens=1, cache=False, temperature=0)
    lm1 = get_huggingface_lm(lm, runtime=runtime)
    out1 = lm1.predict(prompt)
    assert out1.completion_text == "!'"

@pytest.mark.parametrize("runtime", [Runtime.PYTORCH ])
def test_codegen2_predict(runtime):
    lm = "Salesforce/codegen2-1B"
    prompt = LmPrompt("print('Hello world", max_tokens=1, cache=False, temperature=0)
    lm1 = get_huggingface_lm(lm, runtime=runtime)
    out1 = lm1.predict(prompt)
    assert out1.completion_text == "!"


@pytest.mark.parametrize("lm", ["Salesforce/codegen2-1B"])
def test_codegen2_predict_bt(lm):
    with pytest.raises(Exception) as e_info:
        get_huggingface_lm(lm, runtime=Runtime.BETTER_TRANSFORMER)
        assert str(e_info.value).startswith("WARNING BetterTransformer")


@pytest.mark.parametrize("lm", ["Salesforce/codegen2-1B"])
@pytest.mark.skipif(
    CUDA_UNAVAILABLE, reason="Cannot test ORT/ONNX CUDA runtime without CUDA"
)
def test_get_onnx_codegen(lm):
    prompt = LmPrompt("def greet(user): print(f'hello <extra_id_0>!')", max_tokens=5, cache=False, temperature=0)
    lm1 = get_huggingface_lm(lm, runtime=Runtime.ONNX)
    out1 = lm1.predict(prompt)
    assert out1.completion_text == "\n\ndef main():"

@pytest.mark.parametrize("lm", ["Salesforce/codet5p-220m"])
@pytest.mark.skipif(
    CUDA_UNAVAILABLE, reason="Cannot test ORT/ONNX CUDA runtime without CUDA"
)
def test_get_onnx_codet5p(lm):
    prompt = LmPrompt("def greet(user): print(f'hello <extra_id_0>!')", max_tokens=20, cache=False, temperature=0)
    lm1 = get_huggingface_lm(lm, runtime=Runtime.ONNX)
    out1 = lm1.predict(prompt)
    assert out1.completion_text == "args(user):"


# @pytest.mark.parametrize("lm", CAUSAL_MODELS)
# @pytest.mark.skipif(
#     CUDA_UNAVAILABLE, reason="Cannot test ORT/ONNX CUDA runtime without CUDA"
# )
# def test_get_tensorrt(lm: str):
#     get_huggingface_lm(lm, runtime=Runtime.TENSORRT)
