class HuggingFaceModelInfo(str):
    def __new__(
        cls,
        name: str,
        token_limit: int,
        supports_completion: bool = True,
        supports_infill: bool = False,
        supports_instructions: bool = False,
        embedding_model: bool = False,
        gated: bool = False,
    ):
        instance = super().__new__(cls, name)
        # The maximum input token length, usually no. of positional embeddings
        instance._token_limit = token_limit

        # Model supports autoregressive completion
        instance._supports_completion = supports_completion

        # Model supports infilling
        instance._supports_infill = supports_infill

        # Model supports instructions i.e. is instruction tuned / chat model
        instance._supports_instructions = supports_instructions

        # Model is designed to extract features i.e. produce embeddings
        instance._embedding_model = embedding_model

        # Model is gated i.e. requires special access
        instance._gated = gated

        return instance

    @property
    def supports_completion(self):
        return self._supports_completion

    @property
    def supports_infill(self):
        return self._supports_infill

    @property
    def supports_instructions(self):
        return self._supports_instructions

    @property
    def embedding_model(self):
        return self._embedding_model

    @property
    def gated(self):
        return self._gated

    @property
    def token_limit(self):
        return self._token_limit


class _ModelNamesMeta(type):
    def __iter__(cls):
        for attr in cls.__dict__:
            if isinstance(cls.__dict__[attr], HuggingFaceModelInfo):
                yield cls.__dict__[attr]


class HuggingFaceModelNames(metaclass=_ModelNamesMeta):
    # CodeT5+ Family
    CodeT5plus_110M = HuggingFaceModelInfo(
        "Salesforce/codet5p-110m-embedding",
        token_limit=512,
        supports_completion=False,
        embedding_model=True,
    )
    CodeT5plus_110M_Bimodal = HuggingFaceModelInfo(
        "Salesforce/codet5p-220m-bimodal",
        token_limit=512,
        supports_completion=False,
        embedding_model=True,
    )
    CodeT5plus_220M = HuggingFaceModelInfo(
        "Salesforce/codet5p-220m",
        token_limit=512,
    )
    CodeT5plus_220M_Python = HuggingFaceModelInfo(
        "Salesforce/codet5p-220m-py",
        token_limit=512,
    )
    CodeT5plus_770M = HuggingFaceModelInfo(
        "Salesforce/codet5p-770m",
        token_limit=512,
    )
    CodeT5plus_770M_Python = HuggingFaceModelInfo(
        "Salesforce/codet5p-770m-py",
        token_limit=512,
    )
    CodeT5plus_2B = HuggingFaceModelInfo(
        "Salesforce/codet5p-2b",
        token_limit=2_048,
    )
    CodeT5plus_6B = HuggingFaceModelInfo(
        "Salesforce/codet5p-6b",
        token_limit=2_048,
    )
    CodeT5plus_6B = HuggingFaceModelInfo(
        "Salesforce/codet5p-16b",
        token_limit=2_048,
    )
    InstructCodeT5plus_16B = HuggingFaceModelInfo(
        "Salesforce/instructcodet5p-16b",
        token_limit=2_048,
        supports_instructions=True,
    )

    # CodeGen2 Family
    CodeGen2_1B = HuggingFaceModelInfo(
        "Salesforce/codegen2-1B",
        token_limit=2_048,
        supports_infill=True,
    )
    CodeGen2_3_7B = HuggingFaceModelInfo(
        "Salesforce/codegen2-3_7B",
        token_limit=2_048,
        supports_infill=True,
    )
    CodeGen2_7B = HuggingFaceModelInfo(
        "Salesforce/codegen2-7B",
        token_limit=2_048,
        supports_infill=True,
    )
    CodeGen2_16B = HuggingFaceModelInfo(
        "Salesforce/codegen2-16B",
        token_limit=2_048,
        supports_infill=True,
    )

    # CodeGen2.5 Family
    CodeGen25_7B_Python = HuggingFaceModelInfo(
        "Salesforce/codegen25-7b-mono",
        token_limit=2_048,
        supports_infill=True,
    )
    CodeGen25_7B_Multi = HuggingFaceModelInfo(
        "Salesforce/codegen25-7b-multi",
        token_limit=2_048,
        supports_infill=True,
    )
    CodeGen25_7B_Instruct = HuggingFaceModelInfo(
        "Salesforce/codegen25-7b-instruct",
        token_limit=2_048,
        supports_infill=True,
        supports_instructions=True,
    )

    # CodeLLama Family
    # 7B
    CodeLLama_7B_Base = HuggingFaceModelInfo(
        "codellama/CodeLlama-7b-hf",
        token_limit=16_384,
        supports_infill=True,
    )

    CodeLLama_7B_Python = HuggingFaceModelInfo(
        "codellama/CodeLlama-7b-Python-hf",
        token_limit=16_384,
    )
    CodeLLama_7B_Instruct = HuggingFaceModelInfo(
        "codellama/CodeLlama-7b-Instruct-hf",
        token_limit=16_384,
        supports_infill=True,
        supports_instructions=True,
    )
    # 13B
    CodeLLama_13B_Base = HuggingFaceModelInfo(
        "codellama/CodeLlama-13b-hf",
        token_limit=16_384,
        supports_infill=True,
    )
    CodeLLama_13B_Python = HuggingFaceModelInfo(
        "codellama/CodeLlama-13b-Python-hf",
        token_limit=16_384,
    )
    CodeLLama_13B_Instruct = HuggingFaceModelInfo(
        "codellama/CodeLlama-13b-Instruct-hf",
        token_limit=16_384,
        supports_infill=True,
        supports_instructions=True,
    )
    # 34B
    CodeLLama_34B_Base = HuggingFaceModelInfo(
        "codellama/CodeLlama-34b-hf",
        token_limit=16_384,
    )
    CodeLLama_34B_Python = HuggingFaceModelInfo(
        "codellama/CodeLlama-34b-Python-hf",
        token_limit=16_384,
    )
    CodeLLama_34B_Instruct = HuggingFaceModelInfo(
        "codellama/CodeLlama-34b-Instruct-hf",
        token_limit=16_384,
        supports_instructions=True,
    )

    # LLama family
    LLama_2_7B = HuggingFaceModelInfo(
        "meta-llama/Llama-2-7b-hf",
        token_limit=4_096,
        gated=True,
    )
    LLama_2_7B_Chat = HuggingFaceModelInfo(
        "meta-llama/Llama-2-7b-chat-hf",
        token_limit=4_096,
        supports_instructions=True,
        gated=True,
    )
    LLama_2_13B = HuggingFaceModelInfo(
        "meta-llama/Llama-2-13b-hf",
        token_limit=4_096,
        gated=True,
    )
    LLama_2_13B_Chat = HuggingFaceModelInfo(
        "meta-llama/Llama-2-13b-chat-hf",
        token_limit=4_096,
        supports_instructions=True,
        gated=True,
    )
    LLama_2_70B = HuggingFaceModelInfo(
        "meta-llama/Llama-2-70b-hf",
        token_limit=4_096,
        gated=True,
    )
    LLama_2_70B_Chat = HuggingFaceModelInfo(
        "meta-llama/Llama-2-70b-chat-hf",
        token_limit=4_096,
        supports_instructions=True,
        gated=True,
    )

    # GPT2 Family
    GPT2 = HuggingFaceModelInfo(
        "gpt2",
        token_limit=1_024,
        supports_instructions=True,
    )
    DistilGPT2 = HuggingFaceModelInfo(
        "distilgpt2",
        token_limit=1_024,
        supports_instructions=True,
    )

    # Zephyr
    Zephyr_7B_Beta = HuggingFaceModelInfo(
        "HuggingFaceH4/zephyr-7b-beta",
        token_limit=32_768,
        supports_instructions=True,
    )

    # Falcon
    Falcon_40B = HuggingFaceModelInfo(
        "tiiuae/falcon-40b-instruct",
        token_limit=2_048,
        supports_instructions=True,
    )

    # Xwin
    Xwin_7B_V2 = HuggingFaceModelInfo(
        "Xwin-LM/Xwin-LM-7B-V0.2",
        token_limit=4_096,
        supports_instructions=True,
    )
    Xwin_13B_V2 = HuggingFaceModelInfo(
        "Xwin-LM/Xwin-LM-13B-V0.2",
        token_limit=4_096,
        supports_instructions=True,
    )

    @classmethod
    def name_to_info(cls, name: str) -> HuggingFaceModelInfo | None:
        if isinstance(name, HuggingFaceModelInfo):
            return name
        for info in cls:
            if info == name:
                return info
        return None
