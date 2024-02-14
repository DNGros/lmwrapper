import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, StoppingCriteria


class _StringStroppingCriteria(StoppingCriteria):
    def __init__(
        self,
        stop_sequences: list[list[str]] = [],
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = None,
        input_length: int | None = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.input_length = input_length
        if not isinstance(stop_sequences, list) or not all(isinstance(stop_sequence, str) for stop_sequence in stop_sequences):
            msg = "Stop sequences must be a list of strings."
            raise ValueError(msg)

        if not isinstance(tokenizer, (PreTrainedTokenizerFast, PreTrainedTokenizer)):
            msg = "Tokenizer must be a `PreTrainedTokenizer` or `PreTrainedTokenizerFast`."

        self.stop_sequences = stop_sequences

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> bool:
        if input_ids.shape[0] != 1:
            msg = "Batches greater than size 1 are not supported."
            raise NotImplementedError(msg)

        input_ids_list = input_ids[0]
        # Decode the string and check for matching string
        decoded_output = self.tokenizer.decode(input_ids_list[self.input_length :])

        return any(
            stop_sequence in decoded_output for stop_sequence in self.stop_sequences
        )

class _TokenIdStoppingCriteria(StoppingCriteria):
    def __init__(
        self,
        stop_sequences: list[list[str]] = [],
        tokenizer: PreTrainedTokenizerFast = None,
        input_length: int | None = None,
    ):
        super().__init__()
        self.tokenizer: PreTrainedTokenizerFast = tokenizer
        self.input_length = input_length
        assert all(isinstance(stop_sequence, str) for stop_sequence in stop_sequences)

        stop_tokens = self._tokenizer(
            stop_sequences,
            add_special_tokens=False,
            return_attention_mask=False,
        ).input_ids
        assert len(stop_tokens) == len(stop_sequences)

        self.stop_sequences = stop_tokens

        assert all(
            isinstance(stop_sequence, int) for stop_sequence in self.stop_sequences
        )

        self.max_stop_sequence_length = max(map(len, stop_sequences))

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> bool:
        if input_ids.shape[0] != 1:
            msg = "Batches greater than size 1 are not supported."
            raise NotImplementedError(msg)

        input_ids_list = input_ids[0]

        # Scan the end of the id sequence
        if self.max_stop_sequence_length > len(input_ids_list):
            span = len(input_ids_list)
        else:
            span = self.max_stop_sequence_length
        input_ids_interest = input_ids_list[-span:]

        return any(
            any(
                input_ids_interest[i : i + len(stop_sequence)] == stop_sequence
                for i in range(len(input_ids_interest) - len(stop_sequence) + 1)
            )
            for stop_sequence in self.stop_sequences
        )