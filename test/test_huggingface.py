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

@pytest.mark.skipif(
    CUDA_UNAVAILABLE, reason="Cannot test ORT/ONNX CUDA runtime without CUDA"
)
def test_onnx_works():
    ort_model = ORTModelForSequenceClassification.from_pretrained(
      "distilbert-base-uncased-finetuned-sst-2-english",
      export=True,
      provider="CUDAExecutionProvider",
    )

    tokenizer = AutoTokenizer.from_pretrained("philschmid/tiny-bert-sst2-distilled")
    inputs = tokenizer("Both the music and visual were astounding, not to mention the actors performance.", return_tensors="pt", padding=True)
    outputs = ort_model(**inputs)
    assert outputs
    assert ort_model.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    pipe = pipeline(task="text-classification", model=ort_model, tokenizer=tokenizer)
    result = pipe("Both the music and visual were astounding, not to mention the actors performance.")
    assert result[0]["label"] == "POSITIVE"

@pytest.mark.parametrize("lm", ALL_MODELS)
def test_get_pytorch(lm):
    get_huggingface_lm(lm, runtime=Runtime.PYTORCH)


@pytest.mark.parametrize("lm", CAUSAL_MODELS)
def test_get_ort_cpu(lm):
    get_huggingface_lm(lm, runtime=Runtime.ORT_CPU)


@pytest.mark.parametrize("lm", ["distilgpt2", "gpt2"])
def test_get_better_transformer(lm):
    if lm.startswith("Salesforce/codegen2"):
        return
    get_huggingface_lm(lm, runtime=Runtime.BETTER_TRANSFORMER)


@pytest.mark.parametrize("lm", ["Salesforce/codegen2-1B"])
def test_codegen2_predict(lm):
    prompt = LmPrompt("print('Hello world", max_tokens=1, cache=False, temperature=0)
    lm1 = get_huggingface_lm(lm, runtime=Runtime.PYTORCH)
    out1 = lm1.predict(prompt)
    assert out1.completion_text == "!"


@pytest.mark.parametrize("lm", ["Salesforce/codegen2-1B"])
def test_codegen2_predict_bt(lm):
    with pytest.raises(Exception) as e_info:
        get_huggingface_lm(lm, runtime=Runtime.BETTER_TRANSFORMER)
        assert str(e_info.value).startswith("WARNING BetterTransformer")


@pytest.mark.parametrize("lm", CAUSAL_MODELS)
@pytest.mark.skipif(
    CUDA_UNAVAILABLE, reason="Cannot test ORT/ONNX CUDA runtime without CUDA"
)
def test_get_onnx(lm):
    get_huggingface_lm(lm, runtime=Runtime.ONNX)


@pytest.mark.parametrize("lm", CAUSAL_MODELS)
@pytest.mark.skipif(
    CUDA_UNAVAILABLE, reason="Cannot test ORT/ONNX CUDA runtime without CUDA"
)
def test_get_tensorrt(lm):
    get_huggingface_lm(lm, runtime=Runtime.TENSORRT)
