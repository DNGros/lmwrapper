from dataclasses import dataclass
from typing import Any

import numpy as np

from lmwrapper.abstract_predictor import LmPredictor
from lmwrapper.structs import LmChatDialog, LmPrediction, LmPrompt

try:
    from exllamav2 import (
        ExLlamaV2,
        ExLlamaV2Cache,
        ExLlamaV2Config,
        ExLlamaV2Tokenizer,
    )
    from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler
except ImportError:
    msg = "Error importing ExLLamaV2. Please verify your installation."
    raise ImportError(
        msg,
    )


@dataclass
class ExLlamaPrediction(LmPrediction):
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


class ExLlamaPredictor(LmPredictor):
    def __init__(
        self,
        generator: ExLlamaV2BaseGenerator,
        tokenizer: ExLlamaV2Tokenizer,
    ):
        self._llm = generator
        self._tokenizer = tokenizer

    def _get_cache_key_metadata(self):
        return {
            "model": "ExLlamaPredictor",
            "name_or_path": self._llm.model.config.model_dir,
        }

    def _predict_maybe_cached(
        self,
        prompt: LmPrompt,
    ) -> LmPrediction | list[LmPrediction]:
        if prompt.n > 1:
            msg = "Multiple completions not supported yet."
            raise ValueError(msg)

        # TODO:
        # collected_outputs = []
        # for b, batch in enumerate(batches):

        #     print(f"Batch {b + 1} of {len(batches)}...")

        #     outputs = generator.generate_simple(batch, settings, max_new_tokens, seed = 1234)

        #     trimmed_outputs = [o[len(p):] for p, o in zip(batch, outputs)]
        #     collected_outputs += trimmed_outputs
        echo_without_generation = prompt.echo and prompt.max_tokens == 0

        sampling_params = SamplingParams(
            n=prompt.num_completions,
            presence_penalty=prompt.presence_penalty,
            frequency_penalty=prompt.frequency_penalty,
            temperature=prompt.temperature,
            top_p=prompt.top_p,
            top_k=prompt.top_k,
            stop=prompt.stop,
            max_tokens=prompt.max_tokens if not echo_without_generation else 1,
            logprobs=prompt.logprobs,
            prompt_logprobs=prompt.logprobs if prompt.echo else None,
            skip_special_tokens=not prompt.add_special_tokens,
        )

        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = prompt.temperature
        settings.top_p = prompt.top_p
        settings.top_k = prompt.top_k
        settings.token_frequency_penalty = prompt.frequency_penalty
        settings.token_presence_penalty = prompt.presence_penalty
        settings.disallow_tokens(self._tokenizer, [self._tokenizer.eos_token_id])

        self._llm.set_stop_conditions([*prompt.stop, self._tokenizer.eos_token_id])
        max_tokens = prompt.max_tokens if not echo_without_generation else 1
        self._llm.generate_simple(prompt, settings, max_tokens, seed=1234)

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


def get_exllama_lm(
    model_directory: str,
) -> ExLlamaPredictor:
    config = ExLlamaV2Config()
    config.model_dir = model_directory
    config.prepare()

    model = ExLlamaV2(config)

    cache = ExLlamaV2Cache(model, lazy=True)
    model.load_autosplit(cache)

    tokenizer = ExLlamaV2Tokenizer(config)

    generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)
    generator.warmup()
    return ExLlamaPredictor(generator)
