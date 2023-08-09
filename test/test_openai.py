import math
import warnings

import pytest

from lmwrapper.caching import clear_cache_dir
from lmwrapper.openai_wrapper import get_open_ai_lm, OpenAiModelNames
from lmwrapper.secret_manage import SecretEnvVar, SecretFile
from lmwrapper.structs import LmPrompt, LmChatDialog

SKIP_OPENAI = not SecretEnvVar("OPENAI_API_KEY").is_readable()

@pytest.mark.skipif(SKIP_OPENAI, reason="No OpenAI key available.")
def play_with_probs():
    lm = get_open_ai_lm()
    out = lm.predict(
        LmPrompt(
            "Once upon",
            max_tokens=2,
            logprobs=5,
            cache=True,
            num_completions=1,
            echo=False
        ))
    print(out)
    print(out._get_completion_token_index())
    print(out.completion_tokens)
    print(out.completion_token_offsets)
    print(out.completion_logprobs)
    print(out.prompt_tokens)


@pytest.mark.skipif(SKIP_OPENAI, reason="No OpenAI key available.")
def test_too_large_logprob():
    """Expect a warning to be thrown when logprobs is greater than 5 (which
    is the limit the openai api supports)"""
    lm = get_open_ai_lm()
    with pytest.warns(None) as called_warnings:
        lm.predict(
            LmPrompt(
                "Once",
                max_tokens=1,
                logprobs=5,
                cache=False,
                num_completions=1,
                echo=False
            )
        )
    assert len(called_warnings) == 0
    with pytest.warns(UserWarning):
        lm.predict(
            LmPrompt(
                "Once",
                max_tokens=1,
                logprobs=10,
                cache=False,
                num_completions=1,
                echo=False
            )
        )



@pytest.mark.skipif(SKIP_OPENAI, reason="No OpenAI key available.")
def test_simple_chat_mode():
    lm = get_open_ai_lm(OpenAiModelNames.gpt_3_5_turbo)
    out = lm.predict(LmPrompt(
        "What is 2+2? Answer with just one number.",
        max_tokens=1,
        num_completions=1,
        cache=False,
    ))
    assert out.completion_text.strip() == "4"


@pytest.mark.skipif(SKIP_OPENAI, reason="No OpenAI key available.")
def test_simple_chat_mode_multiturn():
    lm = get_open_ai_lm(OpenAiModelNames.gpt_3_5_turbo)
    prompt = [
        "What is 2+2? Answer with just one number.",
        "4",
        "What is 3+2?"
    ]
    assert LmChatDialog(prompt).as_dicts() == [
        {'role': 'user', 'content': 'What is 2+2? Answer with just one number.'},
        {'role': 'system', 'content': '4'},
        {'role': 'user', 'content': 'What is 3+2?'}
    ]
    out = lm.predict(LmPrompt(
        prompt,
        max_tokens=1,
        num_completions=1,
        cache=False,
    ))
    assert out.completion_text.strip() == "5"


def main():
    play_with_probs()
    exit()
    lm = get_open_ai_lm()
    clear_cache_dir()
    text = lm.predict(
        LmPrompt(
            "Once upon a time there was a Goose. And",
            max_tokens=1, logprobs=10, cache=True
        ))
    print(text.completion_text)
    lm = get_open_ai_lm()
    new_text = lm.predict(
        LmPrompt(
            "Once upon a time there was a Goose. And",
            max_tokens=1, logprobs=10, cache=True
        ))
    print(new_text.completion_text)
    assert text.completion_text == new_text.completion_text
    exit()
    lm = get_open_ai_lm()
    text = lm.predict(
        LmPrompt(
            "Once upon a time there was a Goose. And",
            max_tokens=1, logprobs=10,
        ))
    print(text.completion_text)

@pytest.mark.skipif(SKIP_OPENAI, reason="No OpenAI key available.")
def test_tokenizer_gpt3():
    lm = get_open_ai_lm(
        OpenAiModelNames.text_ada_001
    )
    assert lm.token_limit == 2049
    assert lm.estimate_tokens_in_prompt(LmPrompt("Once", max_tokens=10)) == 1
    assert lm.estimate_tokens_in_prompt(LmPrompt(
        "Once upon a time there was a magical place with a Spunctulus that was 3281 years old",
        max_tokens=10
    )) == 20
    assert lm.estimate_tokens_in_prompt(LmPrompt(
       " ".join(["once"] * 2000), max_tokens=10
    )) == 2000
    assert not lm.could_completion_go_over_token_limit(LmPrompt(
        " ".join(["once"] * 2000), max_tokens=10
    ))
    assert lm.could_completion_go_over_token_limit(LmPrompt(
        " ".join(["once"] * 2000), max_tokens=50
    ))

@pytest.mark.skipif(SKIP_OPENAI, reason="No OpenAI key available.")
def test_tokenizer_chat():
    lm = get_open_ai_lm(
        OpenAiModelNames.gpt_3_5_turbo
    )
    assert lm.token_limit == 4096
    assert 5 < lm.estimate_tokens_in_prompt(LmPrompt(
        "Once upon a time", max_tokens=10
    )) < 4 + 12


if __name__ == "__main__":
    main()
