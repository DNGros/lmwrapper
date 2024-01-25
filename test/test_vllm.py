import numpy as np
import pytest
import torch

from lmwrapper.vllm.wrapper import get_vllm_lm
from lmwrapper.runtime import Runtime
from lmwrapper.structs import LmPrompt
from lmwrapper.utils import StrEnum

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