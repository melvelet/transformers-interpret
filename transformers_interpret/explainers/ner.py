from typing import List, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers_interpret import BaseExplainer, LIGAttributions
from transformers_interpret.errors import (
    AttributionTypeNotSupportedError,
    InputIdsNotCalculatedError,
)
from transformers_interpret.explainers.question_answering import (
    SUPPORTED_ATTRIBUTION_TYPES,
)

SUPPORTED_ATTRIBUTION_TYPES = ["lig"]


class NERExplainer(BaseExplainer):
    """"""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        attribution_type: str = "lig",
    ):
        super().__init__(model, tokenizer)
        if attribution_type not in SUPPORTED_ATTRIBUTION_TYPES:
            raise AttributionTypeNotSupportedError(
                f"""Attribution type '{attribution_type}' is not supported.
                Supported types are {SUPPORTED_ATTRIBUTION_TYPES}"""
            )
        self.attribution_type = attribution_type

        self.attributions = []

        self.internal_batch_size = None
        self.n_steps = 50

    def __call__(
        self,
        text: str,
        embedding_type: int,
        ignore_tokens: list = [],
        internal_batch_size: int = None,
        n_steps: int = None,
    ) -> dict:

        if n_steps:
            self.n_steps = n_steps
        if internal_batch_size:
            self.internal_batch_size = internal_batch_size

    def encode(self, text: str):
        return super().encode(text=text)

    def decode(self, input_ids: torch.Tensor) -> List[str]:
        return super().decode(input_ids)

    @property
    def word_attributions(self):
        pass

    def _forward(self):
        return super()._forward()

    def _run(self) -> list:
        return super()._run()

    def _calculate_attributions(self):
        return super()._calculate_attributions()


TokenClassificationExplainer = NERExplainer
