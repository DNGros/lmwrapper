from dataclasses import dataclass
from functools import cached_property
from typing import Union, List, Any

from torch import device
from lmwrapper.abstract_predictor import LmPredictor
from lmwrapper.structs import LmPrompt, LmPrediction
from packaging import version
from importlib.metadata import version as import_version
from enum import Enum


try:
    import torch

    assert version.parse(torch.__version__) >= version.parse("2.0")
except ImportError:
    raise ImportError(
        "Expect to work on torch. Please see https://pytorch.org/ for install" " info"
    )
quant_config = None
try:
    import bitsandbytes

    assert version.parse(import_version("bitsandbytes")) >= version.parse("0.41.1")

    from transformers import BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        # load_in_8bit (bool, optional, defaults to False) — This flag is used to enable 8-bit quantization with LLM.int8().
        # load_in_4bit (bool, optional, defaults to False) — This flag is used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from bitsandbytes.
        # llm_int8_threshold (float, optional, defaults to 6) — This corresponds to the outlier threshold for outlier detection as described in LLM.int8() : 8-bit Matrix Multiplication for Transformers at Scale paper: https://arxiv.org/abs/2208.07339 Any hidden states value that is above this threshold will be considered an outlier and the operation on those values will be done in fp16. Values are usually normally distributed, that is, most values are in the range [-3.5, 3.5], but there are some exceptional systematic outliers that are very differently distributed for large models. These outliers are often in the interval [-60, -6] or [6, 60]. Int8 quantization works well for values of magnitude ~5, but beyond that, there is a significant performance penalty. A good default threshold is 6, but a lower threshold might be needed for more unstable models (small models, fine-tuning).
        # llm_int8_skip_modules (List[str], optional) — An explicit list of the modules that we do not want to convert in 8-bit. This is useful for models such as Jukebox that has several heads in different places and not necessarily at the last position. For example for CausalLM models, the last lm_head is kept in its original dtype.
        # llm_int8_enable_fp32_cpu_offload (bool, optional, defaults to False) — This flag is used for advanced use cases and users that are aware of this feature. If you want to split your model in different parts and run some parts in int8 on GPU and some parts in fp32 on CPU, you can use this flag. This is useful for offloading large models such as google/flan-t5-xxl. Note that the int8 operations will not be run on CPU.
        # llm_int8_has_fp16_weight (bool, optional, defaults to False) — This flag runs LLM.int8() with 16-bit main weights. This is useful for fine-tuning as the weights do not have to be converted back and forth for the backward pass.
        # bnb_4bit_compute_dtype (torch.dtype or str, optional, defaults to torch.float32) — This sets the computational type which might be different than the input time. For example, inputs might be fp32, but computation can be set to bf16 for speedups.
        # bnb_4bit_quant_type (str, {fp4, nf4}, defaults to fp4) — This sets the quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types which are specified by fp4 or nf4.
        # bnb_4bit_use_double_quant (bool, optional, defaults to False) — This flag is used for nested quantization where the quantization constants from the first quantization are quantized again.
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
    )
except ImportError:
    print("8/4bit quantization is disabled as bitsandbytes could not be loaded.")

try:
    import transformers

    assert version.parse(transformers.__version__) >= version.parse("4.31.0")

    from transformers import (
        GenerationConfig,
        AutoTokenizer,
        AutoModelForCausalLM,
        PreTrainedModel,
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
        set_seed,
        T5ForConditionalGeneration,
        AutoModelForSeq2SeqLM,
    )

    set_seed(42)
except ImportError:
    raise ImportError(
        "You must install transformers and bitsandbytes to use Huggingface models. "
        "`pip install transformers` and torch"
    )

_ONNX_RUNTIME = True  # os.getenv("onnx").lower() == "true"

if _ONNX_RUNTIME:
    try:
        from optimum import version as optimum_version

        assert version.parse(optimum_version.__version__) >= version.parse("1.11.0")

        from optimum.onnxruntime import (
            ORTModelForCausalLM,
            ORTModelForSeq2SeqLM,
            ORTModel,
        )
        from optimum.bettertransformer import BetterTransformer

        import xformers

        assert version.parse(xformers.__version__) >= version.parse("0.0.20")

        import onnxruntime

        assert version.parse(onnxruntime.__version__) >= version.parse("1.15.1")

        session_options = onnxruntime.SessionOptions()
        # session_options.log_severity_level = 0
    except ImportError:
        raise ImportError(
            "You must install Optimum, ONNX runtime, and Xformers to use"
            " accelerated Huggingface models. `pip install"
            " optimum[onnxruntime-gpu] xformers`"
        )


class Runtime(Enum):
    PYTORCH = 1
    ONNX = 2
    TENSORRT = 3
    ORT_CPU = 4
    BETTER_TRANSFORMER = 5


@dataclass
class HuggingfacePrediction(LmPrediction):
    _prompt_encoding: Any
    _tokens: Any
    _log_probs: Any

    def __post_init__(self):
        assert len(self._prompt_encoding["input_ids"]) == 1
        self._num_prompt_tokens = len(self._prompt_encoding["input_ids"][0])
        if self.prompt.add_bos_token:
            self._num_prompt_tokens -= 1

    @property
    def completion_tokens(self) -> List[str]:
        return self._tokens[self._num_prompt_tokens :]

    @property
    def completion_logprobs(self) -> List[float]:
        self._verify_logprobs()
        return self._log_probs[self._num_prompt_tokens :]

    @property
    def prompt_tokens(self):
        return self._tokens[: self._num_prompt_tokens]

    @property
    def prompt_logprobs(self):
        return self._log_probs[: self._num_prompt_tokens]

    @property
    def full_logprobs(self):
        return self._log_probs

    def get_full_tokens(self):
        return self._tokens


class HuggingfacePredictor(LmPredictor):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast | PreTrainedTokenizer,
        model: PreTrainedModel,
        device: str = "cpu",
    ):
        super().__init__()
        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    def _predict_maybe_cached(
        self, prompt: LmPrompt
    ) -> Union[LmPrediction, List[LmPrediction]]:
        if prompt.stop:
            raise NotImplementedError
        if prompt.presence_penalty:
            raise NotImplementedError
        temperature = prompt.temperature
        if temperature == 0:
            temperature = 1e-9
        assert self._tokenizer.bos_token

        not (
            hasattr(self._model, "use_bettertransformer")
            and self._model.use_bettertransformer is True
        )

        encoded_input = self._tokenizer(
            self._tokenizer.bos_token
            + prompt.text,  # TODO: not sure why bos token is prepended for all models?
            return_tensors="pt",
            padding=True,  # TODO: not all models have a padding token
            truncation=True,
            max_length=self._model.config.max_length,
        ).to(self._device)

        self._model.to(self._device)
        # output = self._model(**encoded_input)
        # text = self._tokenizer.decode(output[0])
        # Ref https://gist.github.com/kinoc/8a042d8c5683725aa8c372274c02ea2f
        need_log_prob = prompt.logprobs is not None and prompt.logprobs > 0

        gen_config = GenerationConfig(
            max_new_tokens=prompt.max_tokens,
            temperature=temperature,
            top_p=prompt.top_p,
            do_sample=prompt.temperature > 0,
            return_dict_in_generate=True,
            output_scores=need_log_prob,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            bos_token_id=self._tokenizer.bos_token_id,
        )

        # We need a way of getting the raw logprobs of the whole sequence.
        #   The scores we get back are possibly already warped by the configuration
        #   https://github.com/huggingface/transformers/issues/17521#issue-1257881647
        #   Also, it does not return the input tokens. Existing hacks
        #   require calling the model again https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/17
        # Instead we are going to patch the model forward to log calls

        old_forward = self._model.forward
        cached_logits = []

        def new_call(*args, **kwargs):
            nonlocal cached_logits
            val = old_forward(*args, **kwargs)
            cached_logits.append(val.logits)
            return val

        self._model.forward = new_call

        with torch.no_grad():
            generation_output = self._model.generate(
                input_ids=encoded_input["input_ids"],
                generation_config=gen_config,
            )

        self._model.forward = old_forward

        s = generation_output.sequences[0]
        text = self._tokenizer.decode(
            s[len(encoded_input["input_ids"][0]) :],
        )

        cleaned_text = self._tokenizer.decode(
            s[len(encoded_input["input_ids"][0]) :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        tokens = self._tokenizer.convert_ids_to_tokens(s)
        # strip the bos token
        tokens = tokens[1:]
        # Calculate the logprobs if needed
        if need_log_prob:
            all_logits = torch.cat(cached_logits, dim=1)
            assert all_logits.shape[0] == 1  # batch
            assert all_logits.shape[1] == len(tokens)
            logprobs = _gather_logprobs_from_logits(
                all_logits[0],
                s[1:],
            )
            assert len(logprobs) == len(tokens)
        else:
            logprobs = None

        if prompt.max_tokens == 0:
            # Huggingface seems to default to one token always return an extra token
            tokens = tokens[:-1]
            logprobs = logprobs[:-1]
            text = ""
            generation_output.sequences = generation_output.sequences[:, :-1]

        return HuggingfacePrediction(
            completion_text=text,
            clean_completion_text=cleaned_text,
            prompt=prompt,
            metad=generation_output,
            _prompt_encoding=encoded_input,
            _tokens=tokens,
            _log_probs=logprobs,
        )

    @cached_property
    def space_char(self) -> str:
        # Try to discover the space char in the tokens
        tokens = self._tokenizer.tokenize("I went to")
        for tok in tokens:
            if "went" in tok:
                val = tok.replace("went", "")
                return val
        return None

    def remove_special_chars_from_tokens(self, tokens: list[str]) -> list[str]:
        if self.space_char is None:
            return tokens
        return [tok.replace(self.space_char, " ") for tok in tokens]


def _gather_logprobs_from_logits(
    logits: torch.Tensor,
    selected_toks: torch.LongTensor,
):
    logprobs = torch.log_softmax(logits, dim=-1).detach()
    gen_probs = torch.gather(logprobs, -1, selected_toks.unsqueeze(-1)).squeeze(-1)
    return gen_probs


def get_accelerator() -> device:
    if torch.cuda.is_available():
        if quant_config:
            assert bitsandbytes.COMPILED_WITH_CUDA
        return torch.device("cuda")

    # if torch.backends.mps.is_available():
    # return torch.device("mps")

    return torch.device("cpu")


def get_huggingface_lm(
    model: str,
    runtime: Runtime = Runtime.PYTORCH,
    precision: torch.dtype = torch.float32,
) -> HuggingfacePredictor:
    _kwargs = {}
    model_class = AutoModelForCausalLM
    if model.startswith("Salesforce/codegen"):
        if runtime == Runtime.BETTER_TRANSFORMER:
            raise Exception(
                "WARNING BetterTransformer breaks CodeGen models with AutoClass. Please use a different model or runtime."
            )
        else:
            _kwargs = {"trust_remote_code": True, "revision": "main"}
    elif model.startswith("Salesforce/codet5") and not model.endswith("b"):
        model_class = T5ForConditionalGeneration
    elif model.startswith("Salesforce/codet5p-") and model.endswith("b"):
        model_class = AutoModelForSeq2SeqLM
        _kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        precision = torch.float16

    return initialize_hf_model(
        model, model_class, runtime=runtime, precision=precision, _kwargs=_kwargs
    )


def get_ort_model(model: PreTrainedModel) -> ORTModel:
    if model in {T5ForConditionalGeneration, AutoModelForSeq2SeqLM}:
        return ORTModelForSeq2SeqLM

    return ORTModelForCausalLM


def initialize_hf_model(
    model_name: str,
    model_class: PreTrainedModel,
    runtime: Runtime = Runtime.PYTORCH,
    precision: torch.dtype = torch.float32,
    _kwargs: dict = {},
) -> HuggingfacePredictor:
    torch_device = get_accelerator()
    match runtime:
        case Runtime.PYTORCH:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            model = model_class.from_pretrained(
                model_name, torch_dtype=precision, **_kwargs
            )

            warmup_model(model, tokenizer, device=torch_device)
        case Runtime.ORT_CPU:
            if not torch_device.type == "cpu":
                print(
                    f"Specified torch device {torch_device} but ORT CPU runtime"
                    " can only use CPU."
                )
            provider_options = {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": f"tmp/trt_cache_{model_name}_cpu",
            }
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = get_ort_model(model_class).from_pretrained(
                model_name,
                export=True,
                provider="CPUExecutionProvider",
                provider_options=provider_options,
                session_options=session_options,
                **_kwargs,
            )
            assert "CPUExecutionProvider" in model.providers

            warmup_model(model, tokenizer, device="cpu")
        case Runtime.ONNX:
            if not torch_device.type == "cuda":
                raise Exception("Cannot run model on CUDA without CUDA.")
            provider_options = {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": f"tmp/trt_cache_{model_name}_onnx",
            }
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            model = get_ort_model(model_class).from_pretrained(
                model_name,
                export=True,
                provider="CUDAExecutionProvider",
                provider_options=provider_options,
                session_options=session_options,
                **_kwargs,
            )
            assert "CUDAExecutionProvider" in model.providers
            warmup_model(model, tokenizer, device=torch_device)
        case Runtime.TENSORRT:
            if not torch_device.type == "cuda":
                raise Exception("Cannot run model on CUDA without CUDA.")
            provider_options = {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": f"tmp/trt_cache_{model_name}_tensorrt",
            }
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = get_ort_model(model_class).from_pretrained(
                model_name,
                export=True,
                provider="TensorrtExecutionProvider",
                provider_options=provider_options,
                session_options=session_options,
                torch_dtype=precision,
                **_kwargs,
            )
            assert "TensorrtExecutionProvider" in model.providers
            warmup_model(model, tokenizer, device=torch_device)
        case Runtime.BETTER_TRANSFORMER:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = BetterTransformer.transform(
                model_class.from_pretrained(
                    model_name, torch_dtype=precision, **_kwargs
                )
            )
            warmup_model(model, tokenizer, device=torch_device)

    return HuggingfacePredictor(tokenizer, model)


def warmup_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    device: device,
):
    text = ["Hello, I'm a language model" * 1]
    if model.config.pad_token_id is None:
        tokenizer.pad_token_id = 0
    encoded_input = tokenizer(
        text,
        return_tensors="pt",
        padding=not (
            hasattr(model, "use_bettertransformer")
            and model.use_bettertransformer is True
        ),  # Don't pad for bettertransformers
        truncation=True,
        max_length=model.config.max_length,
    ).to(device)

    generation_config = GenerationConfig(
        max_new_tokens=20,
        # max_new_tokens=
        do_sample=True,
        num_return_sequences=5,
        num_beams=4,
        # do_sample=True
        pad_token_id=tokenizer.pad_token_id,
    )
    assert len(encoded_input[0]) < model.config.max_length
    model.to(device)
    for i in range(3):
        output = model.generate(**encoded_input, generation_config=generation_config)
        tokenizer.decode(
            output[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
