import warnings
from typing import List, Union

import torch
from torch.nn.modules.sparse import Embedding
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

    def _get_preds(
        self,
        input_ids: torch.Tensor,
        token_type_ids=None,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):
        if self.accepts_position_ids and self.accepts_token_type_ids:
            preds = self.model(
                input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

            return preds

        elif self.accepts_position_ids:
            preds = self.model(
                input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

            return preds
        elif self.accepts_token_type_ids:
            preds = self.model(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )

            return preds
        else:
            preds = self.model(
                input_ids,
                attention_mask=attention_mask,
            )

            return preds

    def _forward(  # type: ignore
        self,
        input_ids: torch.Tensor,
        token_type_ids=None,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):

        preds = self._get_preds(input_ids, token_type_ids, position_ids, attention_mask)
        # print(preds[0].size())
        preds = preds[0][0][self.position]

        return preds.max(1).values

    def _run(self, text: str, embedding_type: int) -> list:
        if embedding_type == 0:
            embeddings = self.word_embeddings
        try:
            if embedding_type == 1:
                if self.accepts_position_ids and self.position_embeddings is not None:
                    embeddings = self.position_embeddings
                else:
                    warnings.warn(
                        "This model doesn't support position embeddings for attributions. Defaulting to word embeddings"
                    )
                    embeddings = self.word_embeddings
            elif embedding_type == 2:
                embeddings = self.model_embeddings

            else:
                embeddings = self.word_embeddings
        except Exception:
            warnings.warn(
                "This model doesn't support the embedding type you selected for attributions. Defaulting to word embeddings"
            )
            embeddings = self.word_embeddings

        self.text = text

        self._calculate_attributions(embeddings)

    def _calculate_attributions(self, embeddings: Embedding):
        (
            self.input_ids,
            self.ref_input_ids,
            self.sep_idx,
        ) = self._make_input_reference_pair(self.text)

        (
            self.position_ids,
            self.ref_position_ids,
        ) = self._make_input_reference_position_id_pair(self.input_ids)

        (
            self.token_type_ids,
            self.ref_token_type_ids,
        ) = self._make_input_reference_token_type_pair(self.input_ids, self.sep_idx)

        self.attention_mask = self._make_attention_mask(self.input_ids)

        reference_tokens = [
            token.replace("Ġ", "") for token in self.decode(self.input_ids)
        ]

        lig = LIGAttributions(
            self._forward,
            embeddings,
            reference_tokens,
            self.input_ids,
            self.ref_input_ids,
            self.sep_idx,
            self.attention_mask,
            position_ids=self.position_ids,
            ref_position_ids=self.ref_position_ids,
            token_type_ids=self.token_type_ids,
            ref_token_type_ids=self.ref_token_type_ids,
            internal_batch_size=self.internal_batch_size,
            n_steps=self.n_steps,
        )

        lig.summarize()
        self.attributions.append(lig)

    def _get_predicted_tokens_indices(self, text: str, ignore_tokens: set):
        included_indices = []
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)

        preds = torch.softmax(outputs[0], dim=1)
        print(preds.size())
        for i, pred in enumerate(preds[0]):
            idx = int(torch.argmax(pred))
            if self.id2label[idx] in ignore_tokens:
                continue
            included_indices.append(i)

        return included_indices

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

        self.included_indices = self._get_predicted_tokens_indices(
            text, set(ignore_tokens)
        )
        print(self.id2label)
        print(self.included_indices)

        tokens = self.encode(text)
        print(tokens)
        for i, token in enumerate(tokens):
            if i not in self.included_indices:
                continue
            self.position = i
            print(i,token)
            # self._run(text, embedding_type)


TokenClassificationExplainer = NERExplainer
