import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from lmwrapper.abstract_predictor import LmPredictor
from lmwrapper.huggingface.predictor import HuggingFacePredictor
from lmwrapper.prompt_trimming import PromptTrimmer
from lmwrapper.runtime import Runtime
from lmwrapper.structs import LmChatDialog, LmPrediction, LmPrompt
from lmwrapper.utils import log_cuda_mem

try:
    from vllm import LLM, SamplingParams

    # assert version.parse(torch.__version__) >= version.parse("2.0")
except ImportError:
    msg = "Error importing vLLM. Please verify your installation."
    raise ImportError(
        msg,
    )


def get_vllm_lm(
    model: str,
    tensor_parallel_size: int,
    precision: Literal["float32", "float16", "half", "bfloat16", "auto"] = "auto",
    trust_remote_code: bool = False,
    prompt_trimmer: PromptTrimmer = None,
) -> HuggingFacePredictor:
    llm = LLM(
        model=model,
        trust_remote_code=trust_remote_code,
        dtype=precision,
        tensor_parallel_size=tensor_parallel_size,
    )
    return vLLMPredictor(llm)


@dataclass
class vLLMPrediction(LmPrediction):
    _tokens: Any
    _log_probs: Any
    _logprobs_dict: dict
    _num_prompt_tokens: int

    def __post_init__(self):
        assert self._num_prompt_tokens
        if self.prompt.logprobs == 0:
            return

        if self.prompt.echo:
            assert len(self._tokens) == len(self._log_probs)
        else:
            assert len(self._tokens[self._num_prompt_tokens :]) == len(self._log_probs)

    @property
    def completion_tokens(self) -> list[str]:
        return self._tokens[self._num_prompt_tokens :]

    @property
    def completion_logprobs(self) -> list[float]:
        self._verify_logprobs()
        if self.prompt.echo:
            return self._log_probs[self._num_prompt_tokens :]
        else:
            return self._log_probs

    @property
    def prompt_tokens(self):
        return self._tokens[: self._num_prompt_tokens]

    @property
    def prompt_logprobs(self):
        if not self.prompt.echo:
            msg = "This property is not available unless the prompt echo is set to True"
            raise ValueError(
                msg,
            )
        self._verify_logprobs()
        return self._log_probs[: self._num_prompt_tokens]

    @property
    def full_logprobs(self):
        return self._log_probs

    def get_full_tokens(self):
        return self._tokens

    @property
    def logprobs_dict(self):
        return self._logprobs_dict


class vLLMPredictor(LmPredictor):
    def __init__(
        self,
        llm: LLM,
    ):
        self._llm = llm

    def _get_cache_key_metadata(self):
        return {
            "model": "vLLMPredictor",
            "name_or_path": self._llm.llm_engine.model_config.model,
        }

    def _predict_maybe_cached(
        self,
        prompt: LmPrompt,
    ) -> LmPrediction | list[LmPrediction]:
        echo_without_generation = prompt.echo and prompt.max_tokens == 0

        sampling_params = SamplingParams(
            n=prompt.num_completions,
            presence_penalty=prompt.presence_penalty,
            frequency_penalty=prompt.frequency_penalty,
            repetition_penalty=prompt.repetition_penalty,
            temperature=prompt.temperature,
            top_p=prompt.top_p,
            top_k=prompt.top_k,
            stop=prompt.stop,
            max_tokens=prompt.max_tokens if not echo_without_generation else 1,
            logprobs=prompt.logprobs,
            prompt_logprobs=prompt.logprobs if prompt.echo else None,
            skip_special_tokens=not prompt.add_special_tokens,
        )

        if prompt.is_dialog():
            prompt_token_ids = self._tokenizer.apply_chat_template(
                prompt.text.as_dicts()
                if isinstance(prompt.text, LmChatDialog)
                else prompt.text,
                add_generation_prompt=True,
                return_tensors="pt",
                add_special_tokens=prompt.add_special_tokens,  # TODO: does this make sense for dialog?
            )
            completions = self._llm.generate(
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        else:
            completions = self._llm.generate(
            prompts=prompt.text,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        def prediction(completion):
            tokens = np.concat([completion.prompt_token_ids, completion.token_ids])
            logprobs = np.concat([completions.prompt_logprobs, completion.logprobs])
            logprobs_dicts = dict(zip(logprobs, tokens, strict=True))
            return vLLMPrediction(
                completion_text=completion.text,
                prompt=prompt,
                _num_prompt_tokens=int(completions.prompt_token_ids),
                _tokens=tokens,
                _log_probs=logprobs,
                _logprobs_dict=logprobs_dicts,
            )

        predictions = [prediction(completion) for completion in completions]

        return predictions if len(prediction) else predictions[0]

