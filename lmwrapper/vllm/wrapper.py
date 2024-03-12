from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal

import numpy as np

from lmwrapper.abstract_predictor import LmPredictor
from lmwrapper.prompt_trimming import PromptTrimmer
from lmwrapper.structs import LmChatDialog, LmPrediction, LmPrompt

try:
    from vllm import LLM, SamplingParams
except ImportError:
    msg = "Error importing vLLM. Please verify your installation."
    # raise ImportError(
    #     msg,
    # )


@dataclass
class vLLMPrediction(LmPrediction):
    _tokens: Any
    _log_probs: Any
    _logprobs_dict: OrderedDict
    _num_prompt_tokens: int

    def __post_init__(self):
        assert self._num_prompt_tokens
        if self.prompt.logprobs == 0:
            return

        # import pdb; pdb.set_trace()
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
        super().__init__()
        self._llm = llm
        # self._tokenizer = tokenizer
        # self._model = model
        # self._device = device
        # self._is_chat_model = False
        # self.runtime = runtime
        # self.allow_patch_model_forward = allow_patch_model_forward
        # self.prompt_trimmer = prompt_trimmer
        # self._tokenizer_already_adds_bos = {}

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
        max_new_tokens = (
            self.default_tokens_generated
            if prompt.max_tokens is None
            else prompt.max_tokens
        )
        sampling_params = SamplingParams(
            n=prompt.num_completions,
            presence_penalty=prompt.presence_penalty,
            frequency_penalty=prompt.frequency_penalty,
            temperature=prompt.temperature,
            top_p=prompt.top_p,
            stop=prompt.stop,
            max_tokens=max_new_tokens if not echo_without_generation else 1,
            logprobs=prompt.logprobs,
            prompt_logprobs=prompt.logprobs if prompt.echo else None,
            skip_special_tokens=not prompt.add_special_tokens,
            include_stop_str_in_output=False,
        )

        if prompt.is_text_a_chat():
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
            tokens = []
            logprobs = []
            logprobs_dicts = dict()

            if prompt.logprobs > 0:
                if prompt.echo:
                    tokens = np.concatenate(
                        [completion.prompt_token_ids, completion.outputs[0].token_ids]
                    )
                    logprobs_dicts = [
                        *completion.prompt_logprobs,
                        *completion.outputs[0].logprobs,
                    ]
                else:
                    # tokens = completion.outputs[0].token_ids
                    tokens = np.concatenate(
                        [completion.prompt_token_ids, completion.outputs[0].token_ids]
                    )
                    logprobs_dicts = completion.outputs[0].logprobs
                token_sequence = np.array([list(d.keys())[0] for d in logprobs_dicts])
                logprob_values = np.array([list(d.values())[0] for d in logprobs_dicts])
                logprobs_dict = []
                probabilities = np.exp(logprob_values)

                assert len(token_sequence) == len(logprob_values)
                assert len(token_sequence) == len(probabilities)

                # Create logprobs dict
                for token, score, probability in zip(
                    token_sequence,
                    logprob_values,
                    probabilities,
                    strict=True,
                ):
                    logprobs_dict.append(
                        {
                            "token": int(token),
                            "repr": repr(self._llm.get_tokenizer().decode(token)),
                            "logit": float(score),
                            "probability": float(probability),
                        },
                    )
            output_tokens = self.remove_special_chars_from_tokens(
                self._llm.get_tokenizer().convert_ids_to_tokens(tokens)
            )

            return vLLMPrediction(
                completion_text=completion.outputs[0].text,
                prompt=prompt,
                _num_prompt_tokens=len(completion.prompt_token_ids),
                _log_probs=logprob_values,
                _logprobs_dict=logprobs_dict,
                _tokens=output_tokens,
                metad=None,
            )

        predictions = [prediction(completion) for completion in completions]

        return predictions if len(predictions) > 1 else predictions[0]

    @cached_property
    def space_char(self) -> str | None:
        # Try to discover the space char in the tokens
        tokens = self._llm.get_tokenizer().tokenize("I went to")
        for tok in tokens:
            if "went" in tok:
                return tok.replace("went", "")
        return None

    @cached_property
    def newline_char(self) -> str | None:
        for attempt in ("I\nI", "a\na"):
            tokens = self._llm.get_tokenizer().tokenize(attempt)
            if len(tokens) != 3:
                continue
            return tokens[1]
        return None

    def remove_special_chars_from_tokens(self, tokens: list[str]) -> list[str]:
        if self.space_char is not None and self.space_char != " ":
            tokens = [tok.replace(self.space_char, " ") for tok in tokens]
        if self.newline_char is not None and self.newline_char != "\n":
            tokens = [tok.replace(self.newline_char, "\n") for tok in tokens]
        return tokens

    def tokenize(self, text: str) -> list[str]:
        return self._llm.get_tokenizer().tokenize(text)

    @property
    def token_limit(self):
        return self._llm.llm_engine.model_config.max_model_len

def get_vllm_lm(
    model: str,
    tensor_parallel_size: int = 4,
    precision: Literal["float32", "float16", "half", "bfloat16", "auto"] = "auto",
    trust_remote_code: bool = False,
    prompt_trimmer: PromptTrimmer = None,
) -> vLLMPredictor:
    llm = LLM(
        model=model,
        trust_remote_code=trust_remote_code,
        dtype=precision,
        tensor_parallel_size=tensor_parallel_size,
    )
    return vLLMPredictor(llm)
