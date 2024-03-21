import warnings

import pytest

from lmwrapper.openai_wrapper import (
    TogetherModelNames,
    get_together_lm,
)
from lmwrapper.structs import LmChatDialog, LmPrompt


def test_with_probs_gpt35():
    lm = get_together_lm(TogetherModelNames.mistral_7b)
    out = lm.predict(
        LmPrompt(
            "Respond with true or false:",
            max_tokens=2,
            logprobs=1,
            cache=False,
            num_completions=1,
            echo=False,
            temperature=1,
        ),
    )


def test_too_large_logprob():
    """
    Expect a warning to be thrown when logprobs is greater than 1 (which
    is the limit the openai api supports)
    """
    lm = get_together_lm(TogetherModelNames.mistral_7b_instruct)
    with warnings.catch_warnings():
        lm.predict(
            LmPrompt(
                "Once",
                max_tokens=1,
                logprobs=1,
                cache=False,
                num_completions=1,
                echo=False,
            ),
        )

    with pytest.warns(UserWarning):
        lm.predict(
            LmPrompt(
                "Once",
                max_tokens=1,
                logprobs=10,
                cache=False,
                num_completions=1,
                echo=False,
            ),
        )


def test_simple_chat_mode():
    lm = get_together_lm(TogetherModelNames.mistral_7b_instruct)
    out = lm.predict(
        LmPrompt(
            "What is 2+2? Answer with just one number.",
            max_tokens=2,
            num_completions=1,
            temperature=0.0,
            cache=False,
        ),
    )
    assert out.completion_text.strip() == "4"


def test_chat_nologprob_exception():
    lm = get_together_lm(TogetherModelNames.mistral_7b_instruct)
    out = lm.predict(
        LmPrompt(
            "What is 2+2? Answer with just one number.",
            max_tokens=1,
            num_completions=1,
            temperature=0.0,
            cache=False,
            logprobs=0,
        ),
    )
    with pytest.raises(
        ValueError,
        match="Response does not contain top_logprobs.",
    ) as exc_info:
        out.top_token_logprobs

    assert type(exc_info.value) is ValueError


def test_simple_chat_mode_multiturn():
    lm = get_together_lm(TogetherModelNames.mistral_7b_instruct)
    prompt = [
        "What is 2+2? Answer with just one number.",
        "4",
        "What is 3+2?",
    ]
    assert LmChatDialog(prompt).as_dicts() == [
        {"role": "user", "content": "What is 2+2? Answer with just one number."},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "What is 3+2?"},
    ]
    out = lm.predict(
        LmPrompt(
            prompt,
            max_tokens=20,
            num_completions=1,
            cache=False,
            temperature=0,
        ),
    )
    assert out.completion_text.strip() == "5"


def test_multiturn_chat_logprobs():
    lm = get_together_lm(TogetherModelNames.mistral_7b_instruct)
    prompt = [
        "What is 2+2? Answer with one number followed by a period.",
        "4.",
        "What is 3+2?",
    ]
    assert LmChatDialog(prompt).as_dicts() == [
        {
            "role": "user",
            "content": "What is 2+2? Answer with one number followed by a period.",
        },
        {"role": "assistant", "content": "4."},
        {"role": "user", "content": "What is 3+2?"},
    ]
    out = lm.predict(
        LmPrompt(
            prompt,
            max_tokens=2,
            num_completions=1,
            temperature=0,
            cache=False,
            logprobs=1,
        ),
    )
    assert out.completion_text.strip() == "5"

    # even at t=0, the logprobs have high variance
    # perhaps anti-reverse engineering measures?

    assert out.completion_logprobs == pytest.approx([-0.2783203, -0.040222168], abs=2)
