import logging
from dataclasses import asdict, dataclass
from typing import Self

from transformers import GenerationConfig, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.generation.beam_constraints import Constraint

from lmwrapper.structs import LmPrompt


@dataclass
class HfGenerationConfigBuilder:
    def __iter__(self) -> dict:
        yield from asdict(self).items()

    max_new_tokens: int | None = None
    """(`int`, *optional*):
        The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
    """
    min_new_tokens: int | None = None
    """(`int`, *optional*):
        The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt.
    """
    early_stopping: bool | str | None = False
    """(`bool` or `str`, *optional*, defaults to `False`):
        Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values:
        `True`, where the generation stops as soon as there are `num_beams` complete candidates; `False`, where an
        heuristic is applied and the generation stops when is it very unlikely to find better candidates;
        `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical
        beam search algorithm).
    """
    max_time: float | None = None
    """(`float`, *optional*):
        The maximum amount of time you allow the computation to run for in seconds. generation will still finish
        the current pass after allocated time has been passed.
    """
    # Parameters that control the generation strategy used
    do_sample: bool = False
    """(`bool`, *optional*, defaults to `False`):
        Whether or not to use sampling ; use greedy decoding otherwise.
    """
    num_beams: int | None = 1
    """(`int`, *optional*, defaults to 1):
        Number of beams for beam search. 1 means no beam search.
    """
    num_beam_groups: int | None = 1
    """(`int`, *optional*, defaults to 1):
        Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
        [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
    """
    penalty_alpha: float | None = None
    """(`float`, *optional*):
        The values balance the model confidence and the degeneration penalty in contrastive search decoding.
    """
    use_cache: bool = True
    """(`bool`, *optional*, defaults to `True`):
        Whether or not the model should use the past last key/values attentions (if applicable to the model) to
        speed up decoding.
    """
    # Parameters for manipulation of the model output logits
    temperature: float | None = 1.0
    """(`float`, *optional*, defaults to 1.0):
        The value used to modulate the next token probabilities.
    """
    top_k: int | None = 50
    """(`int`, *optional*, defaults to 50):
        The number of highest probability vocabulary tokens to keep for top-k-filtering.
    """
    top_p: float | None = 1.0
    """(`float`, *optional*, defaults to 1.0):
        If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to
        `top_p` or higher are kept for generation.
    """
    typical_p: float | None = 1.0
    """(`float`, *optional*, defaults to 1.0):
        Local typicality measures how similar the conditional probability of predicting a target token next is to
        the expected conditional probability of predicting a random token next, given the partial text already
        generated. If set to float < 1, the smallest set of the most locally typical tokens with probabilities that
        add up to `typical_p` or higher are kept for generation. See [this
        paper](https://arxiv.org/pdf/2202.00666.pdf) for more details.
    """
    epsilon_cutoff: float | None = 0.0
    """(`float`, *optional*, defaults to 0.0):
        If set to float strictly between 0 and 1, only tokens with a conditional probability greater than
        `epsilon_cutoff` will be sampled. In the paper, suggested values range from 3e-4 to 9e-4, depending on the
        size of the model. See [Truncation Sampling as Language Model
        Desmoothing](https://arxiv.org/abs/2210.15191) for more details.
    """
    eta_cutoff: float | None = 0.0
    """(`float`, *optional*, defaults to 0.0):
        Eta sampling is a hybrid of locally typical sampling and epsilon sampling. If set to float strictly between
        0 and 1, a token is only considered if it is greater than either `eta_cutoff` or `sqrt(eta_cutoff) *
        exp(-entropy(softmax(next_token_logits)))`. The latter term is intuitively the expected next token
        probability, scaled by `sqrt(eta_cutoff)`. In the paper, suggested values range from 3e-4 to 2e-3,
        depending on the size of the model. See [Truncation Sampling as Language Model
        Desmoothing](https://arxiv.org/abs/2210.15191) for more details.
    """
    diversity_penalty: float | None = 0.0
    """(`float`, *optional*, defaults to 0.0):
        This value is subtracted from a beam's score if it generates a token same as any beam from other group at a
        particular time. Note that `diversity_penalty` is only effective if `group beam search` is enabled.
    """
    repetition_penalty: float | None = 1.0
    """(`float`, *optional*, defaults to 1.0):
        The parameter for repetition penalty. 1.0 means no penalty. See [this
        paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    """
    encoder_repetition_penalty: float | None = 1.0
    """(`float`, *optional*, defaults to 1.0):
        The paramater for encoder_repetition_penalty. An exponential penalty on sequences that are not in the
        original input. 1.0 means no penalty.
    """
    length_penalty: float | None = 1.0
    """(`float`, *optional*, defaults to 1.0):
        Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
        the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
        likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
        `length_penalty` < 0.0 encourages shorter sequences.
    """
    no_repeat_ngram_size: int | None = 0.0
    """(`int`, *optional*, defaults to 0):
        If set to int > 0, all ngrams of that size can only occur once.
    """
    bad_words_ids: list[list[int]] | None = None
    """(`List[List[int]]`, *optional*):
        List of list of token ids that are not allowed to be generated. Check
        [`~generation.NoBadWordsLogitsProcessor`] for further documentation and examples.
    """
    force_words_ids: list[list[int]] | list[list[list[int]]] | None = None
    """(`List[List[int]]` or `List[List[List[int]]]`, *optional*):
        List of token ids that must be generated. If given a `List[List[int]]`, this is treated as a simple list of
        words that must be included, the opposite to `bad_words_ids`. If given `List[List[List[int]]]`, this
        triggers a [disjunctive constraint](https://github.com/huggingface/transformers/issues/14081), where one
        can allow different forms of each word.
    """
    renormalize_logits: bool = False
    """(`bool`, *optional*, defaults to `False`):
        Whether to renormalize the logits after applying all the logits processors or warpers (including the custom
        ones). It's highly recommended to set this flag to `True` as the search algorithms suppose the score logits
        are normalized but some logit processors or warpers break the normalization.
    """
    constraints: list[Constraint] | None = None
    """(`List[Constraint]`, *optional*):
        Custom constraints that can be added to the generation to ensure that the output will contain the use of
        certain tokens as defined by `Constraint` objects, in the most sensible way possible.
    """
    forced_bos_token_id: int | None = None
    """(`int`, *optional*, defaults to `model.config.forced_bos_token_id`):
        The id of the token to force as the first generated token after the `decoder_start_token_id`. Useful for
        multilingual models like [mBART](../model_doc/mbart) where the first generated token needs to be the target
        language token.
    """
    forced_eos_token_id: int | list[int] | None = None
    """(`Union[int, List[int]]`, *optional*, defaults to `model.config.forced_eos_token_id`):
        The id of the token to force as the last generated token when `max_length` is reached. Optionally, use a
        list to set multiple *end-of-sequence* tokens.
    """
    remove_invalid_values: bool | None = None
    """(`bool`, *optional*, defaults to `model.config.remove_invalid_values`):
        Whether to remove possible *nan* and *inf* outputs of the model to prevent the generation method to crash.
        Note that using `remove_invalid_values` can slow down generation.
    """
    exponential_decay_length_penalty: tuple(int, float) | None = None
    """(`tuple(int, float)`, *optional*):
        This Tuple adds an exponentially increasing length penalty, after a certain amount of tokens have been
        generated. The tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates where
        penalty starts and `decay_factor` represents the factor of exponential decay"""
    suppress_tokens: list[int] | None = None
    """(`List[int]`, *optional*):
        A list of tokens that will be suppressed at generation. The `SupressTokens` logit processor will set their
        log probs to `-inf` so that they are not sampled.
    """
    begin_suppress_tokens: list[int] | None = None
    """(`List[int]`, *optional*):
        A list of tokens that will be suppressed at the beginning of the generation. The `SupressBeginTokens` logit
        processor will set their log probs to `-inf` so that they are not sampled.
    """
    forced_decoder_ids: list[list[int]] | None = None
    """(`List[List[int]]`, *optional*):
        A list of pairs of integers which indicates a mapping from generation indices to token indices that will be
        forced before sampling. For example, `[[1, 123]]` means the second generated token will always be a token
        of index 123.
    """
    sequence_bias: dict[tuple[int], float] | None = None
    """(`Dict[Tuple[int], float]`, *optional*)):
        Dictionary that maps a sequence of tokens to its bias term. Positive biases increase the odds of the
        sequence being selected, while negative biases do the opposite. Check
        [`~generation.SequenceBiasLogitsProcessor`] for further documentation and examples.
    """
    guidance_scale: float | None = None
    """(`float`, *optional*):
        The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
        Higher guidance scale encourages the model to generate samples that are more closely linked to the input
        prompt, usually at the expense of poorer quality.
    """
    low_memory: bool | None = None
    """(`bool`, *optional*):
        Switch to sequential topk for contrastive search to reduce peak memory. Used with contrastive search.
    """
    # Parameters that define the output variables of `generate`
    num_return_sequences: int | None = 1
    """(`int`, *optional*, defaults to 1):
        The number of independently computed returned sequences for each element in the batch.
    """
    output_attentions: bool | None = False
    """(`bool`, *optional*, defaults to `False`):
        Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
        tensors for more details.
    """
    output_hidden_states: bool | None = False
    """(`bool`, *optional*, defaults to `False`):
        Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
        more details.
    """
    output_scores: bool | None = False
    """(`bool`, *optional*, defaults to `False`):
        Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
    """
    return_dict_in_generate: bool | None = True
    """(`bool`, *optional*, defaults to `True`):
        Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
    """
    # Special tokens that can be used at generation time
    pad_token_id: int | None = None
    """(`int`, *optional*):
        The id of the *padding* token.
    """
    bos_token_id: int | None = None
    """(`int`, *optional*):
        The id of the *beginning-of-sequence* token.
    """
    eos_token_id: int | list[int] | None = None
    """(`Union[int, List[int]]`, *optional*):
        The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
    """
    # Generation parameters exclusive to encoder-decoder models
    encoder_no_repeat_ngram_size: int | None = 0
    """(`int`, *optional*, defaults to 0):
        If set to int > 0, all ngrams of that size that occur in the `encoder_input_ids` cannot occur in the
        `decoder_input_ids`.
    """
    decoder_start_token_id: int | None = None
    """(`int`, *optional*):
        If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token.
    """

    @classmethod
    def from_prompt(
        cls,
        prompt: LmPrompt,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    ) -> Self:
        # Temperature doesn't work as you expect
        # If you pass 0 and do_sample = True, HF raises an exception
        # So we turn off sampling if temperature = 0
        do_sample = prompt.temperature > 0
        config = cls(
            max_new_tokens=prompt.max_tokens,  # TODO: add back default
            output_scores=prompt.logprobs is not None and prompt.logprobs > 0,
            do_sample=do_sample,
            temperature=prompt.temperature if do_sample else None,
            pad_token_id=(
                None if tokenizer.pad_token_id is None else tokenizer.pad_token_id
            ),
            eos_token_id=(
                None if tokenizer.eos_token_id is None else tokenizer.eos_token_id
            ),
            bos_token_id=(
                None if tokenizer.bos_token_id is None else tokenizer.bos_token_id
            ),
        )

        if config.num_beams == 1 and do_sample is False:
            logging.info("Decoding strategy: greedy decoding")
        elif config.penalty_alpha > 0.0 and config.top_k > 1:
            logging.info("Decoding strategy: contrastive search")
        elif config.num_beams == 1 and do_sample is True:
            logging.info("Decoding strategy: multinomial sampling")
        elif config.num_beams > 1 and do_sample is False:
            logging.info("Decoding strategy: beam-search decoding")
        elif config.num_beams > 1 and do_sample is True:
            logging.info("Decoding strategy: beam-search multinomial sampling")
        elif config.num_beams > 1 and config.num_beam_groups > 1:
            logging.info("Decoding strategy: diverse beam-search")
        else:
            logging.info("Unable to predict decoding strategy!")

        return GenerationConfig(**dict(config))
