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

        self.label2id = model.config.label2id
        self.id2label = model.config.id2label

    def encode(self, text: str = None) -> list:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, input_ids: torch.Tensor) -> list:
        "Decode 'input_ids' to string using tokenizer"
        return self.tokenizer.convert_ids_to_tokens(input_ids[0])

    @property
    def word_attributions(self):
        pass

    def _forward(self):
        return super()._forward()

    def _run(self) -> list:
        return super()._run()

    def _calculate_attributions(self):
        return super()._calculate_attributions()

    def _get_predicted_tokens(self, text: str):
        token_mappings = []
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        tokens = self.decode(inputs["input_ids"])

        preds = torch.softmax(outputs[0], dim=1)
        for i, pred in enumerate(preds[0]):
            idx = int(torch.argmax(pred))
            token_mappings.append((self.id2label[idx],i))
        
        print(token_mappings)

        return token_mappings

    def __call__(
        self,
        text: str,
        embedding_type: int = 0,
        ignore_tokens: list = [],
        internal_batch_size: int = None,
        n_steps: int = None,
    ) -> dict:

        if n_steps:
            self.n_steps = n_steps
        if internal_batch_size:
            self.internal_batch_size = internal_batch_size

        self._get_predicted_tokens(text)


TokenClassificationExplainer = NERExplainer
