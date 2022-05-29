from typing import Dict, List, Optional, Tuple, Union

import torch
from captum.attr import visualization as viz
from transformers import PreTrainedModel, PreTrainedTokenizer

from .sequence_classification import SequenceClassificationExplainer

SUPPORTED_ATTRIBUTION_TYPES = ["lig"]


class TokenClassificationExplainer(SequenceClassificationExplainer):
    """
    Explainer for independently explaining label attributions in a multi-label fashion
    for models of type `{MODEL_NAME}ForSequenceClassification` from the Transformers package.
    Every label is explained independently and the word attributions are a dictionary of labels
    mapping to the word attributions for that label. Even if the model itself is not multi-label
    by the resulting word attributions treat the labels as independent.

    Calculates attribution for `text` using the given model
    and tokenizer. Since this is a multi-label explainer, the attribution calculation time scales
    linearly with the number of labels.

    This explainer also allows for attributions with respect to a particular embedding type.
    This can be selected by passing a `embedding_type`. The default value is `0` which
    is for word_embeddings, if `1` is passed then attributions are w.r.t to position_embeddings.
    If a model does not take position ids in its forward method (distilbert) a warning will
    occur and the default word_embeddings will be chosen instead.
    """

    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            attribution_type="lig",
            custom_labels: Optional[List[str]] = None,
    ):
        super().__init__(model, tokenizer, attribution_type, custom_labels)
        self.labels = []

    @property
    def word_attributions(self) -> dict:
        "Returns the word attributions for model and the text provided. Raises error if attributions not calculated."
        def get_word_attributions(attr):
            return attr.word_attributions if attr is not None else attr

        if self.attributions != []:
            return dict(
                zip(
                    self.labels,
                    [dict(
                        zip(
                            [f"{str(i).zfill(2)} ({self.input_tokens[i + 1]})" for i in range(self.input_length)],
                            [get_word_attributions(attr) for attr in self.attributions[label_i]],
                        )) for label_i, _ in enumerate(self.labels)]
            ))

        else:
            raise ValueError("Attributions have not yet been calculated. Please call the explainer on text first.")

    def visualize(self, html_filepath: str = None, true_class: str = None):
        """
        Visualizes word attributions. If in a notebook table will be displayed inline.

        Otherwise, pass a valid path to `html_filepath` and the visualization will be saved
        as a html file.

        If the true class is known for the text that can be passed to `true_class`

        """
        tokens = [token.replace("Ġ", "") for token in self.decode(self.input_ids)]

        score_viz = [
            self.attributions[i].visualize_attributions(  # type: ignore
                self.pred_probs[i],
                "",  # including a predicted class name does not make sense for this explainer
                "n/a" if not true_class else true_class,  # no true class name for this explainer by default
                f"{i}",
                tokens,
            )
            for i in range(len(self.attributions))
        ]

        html = viz.visualize_text(score_viz)

        new_html_data = html._repr_html_().replace("Predicted Label", "Prediction Score")
        new_html_data = new_html_data.replace("True Label", "n/a")
        html.data = new_html_data

        if html_filepath:
            if not html_filepath.endswith(".html"):
                html_filepath = html_filepath + ".html"
            with open(html_filepath, "w") as html_file:
                html_file.write(html.data)
        return html

    def __call__(
            self,
            text: str,
            embedding_type: int = 0,
            token_index: Optional[Union[int, None]] = None,
            class_index: Optional[Union[int, None]] = None,
            internal_batch_size: int = None,
            n_steps: int = None,
    ) -> dict:
        """
        Calculates attributions for `text` using the model
        and tokenizer given in the constructor. Attributions are calculated for
        every label output in the model.

        This explainer also allows for attributions with respect to a particular embedding type.
        This can be selected by passing a `embedding_type`. The default value is `0` which
        is for word_embeddings, if `1` is passed then attributions are w.r.t to position_embeddings.
        If a model does not take position ids in its forward method (distilbert) a warning will
        occur and the default word_embeddings will be chosen instead.

        Args:
            text (str): Text to provide attributions for.
            embedding_type (int, optional): The embedding type word(0) or position(1) to calculate attributions for. Defaults to 0.
            token_index: (int, optional): Only calculate attribution scores for a specified token. Iterate over all as default.
            class_index: (int, optional): Only calculate attribution scores for a specified class. Iterate over all as default.
            internal_batch_size (int, optional): Divides total #steps * #examples
                data points into chunks of size at most internal_batch_size,
                which are computed (forward / backward passes)
                sequentially. If internal_batch_size is None, then all evaluations are
                processed in one batch.
            n_steps (int, optional): The number of steps used by the approximation
                method. Default: 50.

        Returns:
            dict: A dictionary of label to list of attributions.
        """
        if n_steps:
            self.n_steps = n_steps
        if internal_batch_size:
            self.internal_batch_size = internal_batch_size

        self.attributions = []
        self.pred_probs = []
        self.labels = list(self.label2id.keys())
        self.label_probs_dict = {}
        self.input_tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
        self.input_length = len(self.input_tokens) - 2
        explainer = None

        for label_j in range(self.model.config.num_labels):
            self.attributions.append([])
            self.label_probs_dict[self.id2label[label_j]] = []

            for token_i in range(self.input_length):
                if token_index is None or token_i == token_index:
                    if class_index is None or label_j == class_index:
                        print('class', label_j, 'token', token_i)
                        explainer = SequenceClassificationExplainer(
                            model=self.model,
                            tokenizer=self.tokenizer,
                            attribution_type=self.attribution_type,
                            token_index=[token_i],
                        )
                        explainer(text, label_j, None, embedding_type)

                        self.attributions[label_j].append(explainer.attributions)
                        self.pred_probs.append(explainer.pred_probs)
                        self.label_probs_dict[self.id2label[label_j]].append(explainer.pred_probs)
                else:
                    self.attributions[label_j].append(None)
                    self.pred_probs.append(None)
                    self.label_probs_dict[self.id2label[label_j]].append(None)

        self.input_ids = explainer.input_ids if explainer is not None else None
        return self.word_attributions

    def _forward(  # type: ignore
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
    ):

        if self.accepts_position_ids:
            preds = self.model(
                input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            preds = preds[0]

        else:
            preds = self.model(input_ids, attention_mask)[0]

        # print('here', preds.shape)

        # if it is a single output node
        if len(preds[0]) == 1:
            self._single_node_output = True
            self.pred_probs = torch.sigmoid(preds)[0][0]
            return torch.sigmoid(preds)[:, :]

        self.pred_probs = torch.softmax(preds, dim=1)[0][self.token_index, self.selected_index]
        # print('shape', self.pred_probs)
        return torch.softmax(preds, dim=1)[:, self.token_index, self.selected_index]

    def __str__(self):
        s = f"{self.__class__.__name__}("
        s += f"\n\tmodel={self.model.__class__.__name__},"
        s += f"\n\ttokenizer={self.tokenizer.__class__.__name__},"
        s += f"\n\tattribution_type='{self.attribution_type}',"
        s += f"\n\tcustom_labels={self.custom_labels},"
        s += ")"

        return s
