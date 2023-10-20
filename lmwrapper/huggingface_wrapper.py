from enum import Enum
from importlib.metadata import version as import_version
from pathlib import Path
import logging
from lmwrapper.utils import log_cuda_mem

from packaging import version
from lmwrapper.HuggingfacePredictor import HuggingfacePredictor

from lmwrapper.prompt_trimming import PromptTrimmer
from lmwrapper.structs import LmPrompt
from lmwrapper.env import _QUANTIZATION_ENABLED, _ONNX_RUNTIME, _MPS_ENABLED

try:
    import torch

    assert version.parse(torch.__version__) >= version.parse("2.0")
except ImportError:
    msg = "Expect to work on torch. Please see https://pytorch.org/ for install info."
    raise ImportError(
        msg,
    )

if _QUANTIZATION_ENABLED:
    try:
        import bitsandbytes

        assert version.parse(import_version("bitsandbytes")) >= version.parse(
            "0.41.1",
        )

        from transformers import BitsAndBytesConfig

        _QUANT_CONFIG = BitsAndBytesConfig(
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
        logging.warning(
            "8/4bit quantization is disabled as bitsandbytes could not be loaded.",
        )

try:
    import transformers

    assert version.parse(transformers.__version__) >= version.parse("4.33.2")

    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        PretrainedConfig,
        PreTrainedModel,
        PreTrainedTokenizerFast,
        T5ForConditionalGeneration,
        set_seed,
    )

    set_seed(42)
except ImportError:
    msg = (
        "You must install torch and transformers to use Huggingface models."
        " `pip install lmwrapper[huggingface]`. Please see https://pytorch.org/"
        " for install info."
    )
    raise ImportError(
        msg,
    )

if _ONNX_RUNTIME:
    try:
        from optimum import version as optimum_version

        assert version.parse(optimum_version.__version__) >= version.parse(
            "1.11.0",
        )

        import xformers

        assert version.parse(xformers.__version__) >= version.parse("0.0.20")

        import onnxruntime
        from optimum.bettertransformer import BetterTransformer
        from optimum.onnxruntime import (
            ORTModel,
            ORTModelForCausalLM,
            ORTModelForSeq2SeqLM,
            ORTOptimizer,
        )
        from optimum.onnxruntime.configuration import AutoOptimizationConfig

        assert version.parse(onnxruntime.__version__) >= version.parse("1.15.1")

        session_options = onnxruntime.SessionOptions()
        # session_options.log_severity_level = 0 TODO: set configurable log level
    except ImportError:
        msg = (
            "You must install Optimum, ONNX runtime, and Xformers to use"
            " accelerated Huggingface models. `pip install lmwrapper[ort-gpu]`"
        )
        raise ImportError(
            msg,
        )


class Runtime(Enum):
    PYTORCH = 1
    ACCELERATE = 2
    ORT_CUDA = 3
    ORT_TENSORRT = 4
    ORT_CPU = 5
    BETTER_TRANSFORMER = 6


def _get_accelerator() -> torch.device:
    """
    Returns the most suitable device (accelerator) for PyTorch operations.

    Returns:
    --------
    torch.device
        The device determined to be the most suitable for PyTorch operations. One of 'cuda', 'mps', or 'cpu'.

    Raises:
    -------
    AssertionError:
        If CUDA & quantization are enabled but `bitsandbytes` is not compiled with CUDA support.

    Notes:
    ------
    * CUDA is prioritized if available.
    * MPS (Metal Performance Shaders) is used if `_MPS_ENABLED` is True and MPS backend is available. MacOS only.
    * If none of the above, CPU is used.

    Examples:
    ---------
    >>> device = _get_accelerator()
    >>> print(device)
    cuda # or mps or cpu

    """
    if torch.cuda.is_available():
        if _QUANT_CONFIG:
            # If quantization is enabled and bits and bytes is not
            # compiled with CUDA, things don't work right
            assert bitsandbytes.COMPILED_WITH_CUDA
        return torch.device("cuda")

    if _MPS_ENABLED and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_huggingface_lm(
    model: str,
    runtime: Runtime = Runtime.PYTORCH,
    precision: torch.dtype = torch.float32,
    trust_remote_code: bool = False,
    allow_patch_model_forward: bool = True,
    prompt_trimmer: PromptTrimmer = None,
    device: torch.device | str = None,
) -> HuggingfacePredictor:
    """
    Initialize and return a HuggingFace language model for prediction.

    Parameters:
    -----------
    model : str
        The identifier of the pre-trained model to load from the Huggingface Model Hub.

    runtime : Runtime, optional
        The backend to use for inference. Default is `Runtime.PYTORCH`.
        The only currently supported option is `Runtime.PYTORCH`.

    precision : torch.dtype, optional
        The floating-point precision of the model weights. Default is `torch.float32`.

    trust_remote_code : bool, optional
        Whether to trust or run the remote code from the loaded model. Default is False.

    allow_patch_model_forward : bool, optional
        Allows patching of the `forward` method of the model to compute logprobs. Default is True.
        This option is required logprobs are required for unconditional generation.

    prompt_trimmer : PromptTrimmer, optional
        An object that trims the input prompt to fit the model input size. Default is None.

    device : torch.device | str, optional
        The device on which to run the model. Defaults to the system's best available device.

    Returns:
    --------
    HuggingfacePredictor
        An initialized Huggingface model ready for prediction.

    Raises:
    -------
    ValueError:
        If an empty string is provided for `device`
    NotImplementedError:
        If the specified `runtime` is not yet supported.

    Notes:
    ------
    * The `trust_remote_code` option should only be enabled if you have verified and trust the remote code.
    * The function supports different types of models based on the `model` identifier and adjusts settings automatically.

    Examples:
    ---------
    >>> predictor = get_huggingface_lm("gpt-2")
    >>> predictor = get_huggingface_lm("gpt-2", precision=torch.float16, device="cuda:0")
    """
    if isinstance(device, str):
        if device.strip() == "":
            raise ValueError("Empty string provided for device.")
        else:
            device = torch.device(device)

    if runtime != Runtime.PYTORCH:
        msg = (
            "Accelerated inference model support is still under"
            " development. Please use Runtime.PYTORCH until support matures."
        )
        raise NotImplementedError(
            msg,
        )

    _kwargs = {"trust_remote_code": trust_remote_code}

    model_class = AutoModelForCausalLM
    model_config = PretrainedConfig.from_pretrained(model)

    has_remote_code = (
        "auto_map" in model_config and "AutoConfig" in model_config["auto_map"]
    )
    has_vocab_size = ("vocab_size" in model_config)
    has_decoder = ("decoder" in model_config)
    has_decoder_vocab_size = has_decoder and "vocab_size" in model_config.decoder

    # Addresses a bug in Transformers
    # Model transitions i.e. logprobs cannot be calculated if
    # the model config does not have a `vocab_size`
    # We check if the decoder has vocab size and update the config.
    if not has_vocab_size and has_decoder_vocab_size:
        model_config.vocab_size = model_config.decoder.vocab_size


    if not trust_remote_code and has_remote_code:
        msg = (
            "The model provided has remote code and likely will not work as"
            " expected. Please call with `trust_remote_code = True` If you have"
            " read and trust the code."
        )
        raise Exception(
            msg,
        )

    if ("auto_map" in model_config) and ("AutoModelForSeq2SeqLM" in model_config["auto_map"]):
        model_class = AutoModelForSeq2SeqLM

    if model.startswith("Salesforce/codegen"):
        if runtime == Runtime.BETTER_TRANSFORMER:
            msg = (
                "WARNING BetterTransformer breaks CodeGen models with"
                " AutoClass. Please use a different model or runtime."
            )
            raise Exception(
                msg,
            )

        _kwargs |= {
            "revision": "main",
            "use_cache": False,
        }
    elif model.startswith("Salesforce/codet5") and not model.endswith("b"):
        model_class = T5ForConditionalGeneration

        # T5 class does not support this arg,
        # only autoclasses do
        _kwargs.pop("trust_remote_code", None)
    elif model.startswith("Salesforce/codet5p-") and model.endswith("b"):
        model_class = AutoModelForSeq2SeqLM
        _kwargs |= {
            "low_cpu_mem_usage": True,
        }
    elif model == "Salesforce/instructcodet5p-16b":
        model_class = AutoModelForSeq2SeqLM
        _kwargs |= {
            "low_cpu_mem_usage": True,
        }

    return _initialize_hf_model(
        model_name=model,
        model_class=model_class,
        model_config=model_config,
        runtime=runtime,
        precision=precision,
        allow_patch_model_forward=allow_patch_model_forward,
        prompt_trimmer=prompt_trimmer,
        device=device,
        _kwargs=_kwargs,
    )


def get_ort_model(model: PreTrainedModel) -> "ORTModel":
    if model in {T5ForConditionalGeneration, AutoModelForSeq2SeqLM}:
        return ORTModelForSeq2SeqLM

    return ORTModelForCausalLM


def get_huggingface_predictor(
    tokenizer: PreTrainedTokenizerFast,
    model: PreTrainedModel,
    device: torch.device,
    runtime: Runtime = Runtime.PYTORCH,
    allow_patch_model_forward: bool = False,
    prompt_trimmer: PromptTrimmer | None = None,
) -> HuggingfacePredictor:
    return HuggingfacePredictor(
        tokenizer,
        model,
        device=device,
        runtime=runtime,
        allow_patch_model_forward=allow_patch_model_forward,
        prompt_trimmer=prompt_trimmer,
    )


def _initialize_hf_model(
    model_name: str,
    model_class: PreTrainedModel,
    model_config: PretrainedConfig,
    runtime: Runtime = Runtime.PYTORCH,
    precision: torch.dtype | str = "auto",
    allow_patch_model_forward: bool = True,
    prompt_trimmer: PromptTrimmer = None,
    device: torch.device = None,
    _kwargs: dict = {},
) -> HuggingfacePredictor:
    torch_device = _get_accelerator() if device is None else device

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if not tokenizer.is_fast:
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

    if runtime == Runtime.PYTORCH:
        logging.debug("Before model instantiation")
        log_cuda_mem()
        model = model_class.from_pretrained(
            pretrained_model_name_or_path=model_name,
            config=model_config,
            torch_dtype=precision,
            **_kwargs,
        )
        logging.debug("Post model instantiation")
        log_cuda_mem()
    elif runtime == Runtime.ACCELERATE:
        model = model_class.from_pretrained(
            pretrained_model_name_or_path=model_name,
            config=model_config,
            torch_dtype=precision,
            device_map="auto",
            **_kwargs,
        )
    elif runtime == Runtime.ORT_CPU:
        if torch_device.type != "cpu":
            logging.warn(
                f"Specified torch device {torch_device} but ORT CPU runtime"
                " can only use CPU. Please specify device='cpu'.",
            )

        # ORT models do not support these flags
        _kwargs.pop("low_cpu_mem_usage", None)
        _kwargs.pop("device_map", None)

        save_dir = f"{model_name.replace('/','_')}_optimized_cpu_o3"

        if not Path(save_dir).exists():
            model = get_ort_model(model_class).from_pretrained(
                pretrained_model_name_or_path=model_name,
                config=model_config,
                export=True,
                provider="CPUExecutionProvider",
                session_options=session_options,
                **_kwargs,
            )
            assert "CPUExecutionProvider" in model.providers
            optimizer = ORTOptimizer.from_pretrained(model)
            optimization_config = AutoOptimizationConfig.O3()
            optimizer.optimize(
                save_dir=save_dir,
                optimization_config=optimization_config,
            )
        model = get_ort_model(model_class).from_pretrained(
            save_dir,
            provider="CPUExecutionProvider",
        )
    elif runtime == Runtime.ORT_CUDA:
        if torch_device.type != "cuda":
            msg = (
                "Cannot run model on CUDA without CUDA. Please specify"
                " device='cuda'."
            )
            raise Exception(
                msg,
            )

        # ORT models do not support these flags
        _kwargs.pop("low_cpu_mem_usage", None)
        _kwargs.pop("device_map", None)

        save_dir = f"{model_name.replace('/','_')}_optimized_gpu_o3"

        if not Path(save_dir).exists():
            model = get_ort_model(model_class).from_pretrained(
                pretrained_model_name_or_path=model_name,
                config=model_config,
                export=True,
                provider="CUDAExecutionProvider",
                session_options=session_options,
                **_kwargs,
            )
            assert "CUDAExecutionProvider" in model.providers
            optimizer = ORTOptimizer.from_pretrained(model)
            optimization_config = AutoOptimizationConfig.O3()
            optimizer.optimize(
                save_dir=save_dir,
                optimization_config=optimization_config,
            )
        model = get_ort_model(model_class).from_pretrained(
            save_dir,
            provider="CUDAExecutionProvider",
        )
    elif runtime == Runtime.ORT_TENSORRT:
        if torch_device.type != "cuda":
            msg = (
                "Cannot run model on CUDA without CUDA. Please specify"
                " device='cuda'."
            )
            raise Exception(
                msg,
            )

        provider_options = {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": (
                f"tmp/trt_cache_{model_name.replace('/','_')}_tensorrt"
            ),
        }

        # TensorRT models do not support these flags
        _kwargs.pop("low_cpu_mem_usage", None)
        _kwargs.pop("device_map", None)

        model = get_ort_model(model_class).from_pretrained(
            pretrained_model_name_or_path=model_name,
            config=model_config,
            export=True,
            provider="TensorrtExecutionProvider",
            provider_options=provider_options,
            session_options=session_options,
            **_kwargs,
        )
        assert "TensorrtExecutionProvider" in model.providers
    elif runtime == Runtime.BETTER_TRANSFORMER:
        model = BetterTransformer.transform(
            model_class.from_pretrained(
                pretrained_model_name_or_path=model_name,
                config=model_config,
                device_map="auto",
                torch_dtype=precision,
                **_kwargs,
            ),
        )
    else:
        msg = "Invalid Runtime provided."
        raise Exception(msg)

    # Some models do not have a pad token, default to 0
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = 0
        logging.warning(
            "Tokenizer does not have a pad_token_id. Setting pad_token_id to 0. May cause unexpected behavior."
        )

    predictor = get_huggingface_predictor(
        tokenizer=tokenizer,
        model=model,
        device=torch_device,
        runtime=runtime,
        allow_patch_model_forward=allow_patch_model_forward,
        prompt_trimmer=prompt_trimmer,
    )

    if runtime == Runtime.ORT_TENSORRT:
        # Warm up TensorRT model once instantiated.
        logging.info("Warmimg up TensorRT model.")
        _warmup_model(predictor)
        logging.info("Warmup successful.")

    return predictor


def _warmup_model(predictor: HuggingfacePredictor):
    raise NotImplementedError("Model warmup is not support yet.")
    small_prompt = LmPrompt("!", cache=False, temperature=0, max_tokens=1)
    predictor.predict(small_prompt)

    single_token = predictor.tokenize("Hello")[0]
    long_prompt_str = single_token * (predictor.token_limit - 1)
    assert predictor.tokenize(long_prompt_str) == (predictor.token_limit - 1)
    long_prompt = LmPrompt(long_prompt_str, cache=False, temperature=0, max_tokens=1)
    predictor.predict(long_prompt)
