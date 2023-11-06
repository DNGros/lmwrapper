import pytest
import torch

from lmwrapper.huggingface.models import HuggingFaceModelNames
from lmwrapper.huggingface_wrapper import get_huggingface_lm
from lmwrapper.runtime import Runtime
from lmwrapper.structs import LmPrompt

ALL_MODELS = {
    HuggingFaceModelNames.CodeGen25_7B_Instruct,
    HuggingFaceModelNames.CodeGen25_7B_Multi,
    HuggingFaceModelNames.CodeGen25_7B_Python,
    HuggingFaceModelNames.CodeGen2_16B,
    HuggingFaceModelNames.CodeGen2_7B,
    HuggingFaceModelNames.CodeGen2_3_7B,
    HuggingFaceModelNames.CodeGen2_1B,
    HuggingFaceModelNames.CodeLLama_7B_Base,
    HuggingFaceModelNames.CodeLLama_7B_Python,
    HuggingFaceModelNames.CodeLLama_7B_Instruct,
    HuggingFaceModelNames.CodeLLama_13B_Base,
    HuggingFaceModelNames.CodeLLama_13B_Python,
    HuggingFaceModelNames.CodeLLama_13B_Instruct,
    HuggingFaceModelNames.CodeLLama_34B_Base,
    HuggingFaceModelNames.CodeLLama_34B_Python,
    HuggingFaceModelNames.CodeLLama_34B_Instruct,
    HuggingFaceModelNames.CodeT5plus_110M,
    HuggingFaceModelNames.CodeT5plus_220M,
    HuggingFaceModelNames.CodeT5plus_220M_Python,
    HuggingFaceModelNames.CodeT5plus_770M,
    HuggingFaceModelNames.CodeT5plus_770M_Python,
    HuggingFaceModelNames.CodeT5plus_2B,
    HuggingFaceModelNames.CodeT5plus_6B,
    HuggingFaceModelNames.CodeT5plus_16B,
    HuggingFaceModelNames.InstructCodeT5plus_16B,
    HuggingFaceModelNames.Zephyr_7B_Beta,
    HuggingFaceModelNames.Falcon_40B,
    HuggingFaceModelNames.Xwin_7B_V2,
    HuggingFaceModelNames.Xwin_13B_V2,
    HuggingFaceModelNames.LLama_2_7B,
    HuggingFaceModelNames.LLama_2_7B_Chat,
    HuggingFaceModelNames.LLama_2_13B,
    HuggingFaceModelNames.LLama_2_13B_Chat,
    HuggingFaceModelNames.LLama_2_70B,
    HuggingFaceModelNames.LLama_2_70B_Chat,
}


@pytest.mark.slow()
@pytest.mark.parametrize("model", ALL_MODELS)
def test_all_models(model):
    lm = get_huggingface_lm(
        model,
        runtime=Runtime.ACCELERATE,
        trust_remote_code=True,
        precision=torch.float16,
    )

    prompt = LmPrompt(
        "def fibonacci(",
        cache=False,
        temperature=0,
        add_special_tokens=False,
        add_bos_token=False,
        logprobs=1,
    )

    out = lm.predict(prompt)
    print(model, out.completion_text)
    print(out.completion_logprobs)
