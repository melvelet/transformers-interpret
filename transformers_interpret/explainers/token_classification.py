from typing import Dict, List, Optional, Tuple, Union

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
        """Returns the word attributions for model and the text provided. Raises error if attributions not calculated."""
        def get_word_attributions(attr):
            return attr.word_attributions if attr is not None else attr

        if self.attributions != []:
            return dict(
                zip(
                    self.labels,
                    [[get_word_attributions(attr) for attr in self.attributions[label_i]]
                        for label_i in range(len(self.labels))]
                ))

        else:
            raise ValueError("Attributions have not yet been calculated. Please call the explainer on text first.")

    @property
    def word_attributions_with_strings(self) -> dict:
        """Returns the word attributions for model and the text provided, also showing the token value for increased readability.
        Raises error if attributions not calculated."""
        def get_word_attributions(attr):
            return attr.word_attributions if attr is not None else attr

        if self.attributions != []:
            return dict(
                zip(
                    self.labels,
                    [dict(
                        zip(
                            [f"{str(i).zfill(2)} ({self.input_tokens[i]})" for i in range(self.input_length)],
                            [get_word_attributions(attr) for attr in self.attributions[label_i]],
                        )) for label_i in range(len(self.labels))]
                ))

        else:
            raise ValueError("Attributions have not yet been calculated. Please call the explainer on text first.")

    def visualize(self, token_index: int = None, class_index: int = None, html_filepath: str = None, true_class: str = None):
        """
        Visualizes word attributions. If in a notebook table will be displayed inline.

        Otherwise, pass a valid path to `html_filepath` and the visualization will be saved
        as a html file.

        If the true class is known for the text that can be passed to `true_class`

        """
        tokens = [token.replace("Ġ", "") for token in self.decode(self.input_ids)]

        if token_index is not None:
            print(f'Prediction for token {token_index} ({self.input_tokens[token_index]})')
            score_viz = [
                self.attributions[i][token_index].visualize_attributions(  # type: ignore
                    self.pred_probs[token_index][i],
                    "",  # including a predicted class name does not make sense for this explainer
                    "n/a" if not true_class else true_class,  # no true class name for this explainer by default
                    f"{self.labels[i]}",
                    tokens,
                )
                for i in range(len(self.attributions))
            ]

        elif class_index is not None:
            print(f'Prediction for class {class_index} ({self.labels[class_index]})')
            score_viz = [
                self.attributions[class_index][i].visualize_attributions(  # type: ignore
                    self.pred_probs[i][class_index],
                    self.labels[class_index],  # including a predicted class name does not make sense for this explainer
                    "n/a" if not true_class else true_class,  # no true class name for this explainer by default
                    f"{i} ({self.input_tokens[i + 1]})",
                    tokens,
                )
                for i in range(len(self.attributions[class_index]))
            ]
        else:
            raise Exception('Either a class index or a token index must be specified.')

        html = viz.visualize_text(score_viz)

        new_html_data = html._repr_html_()
        if token_index is not None:
            new_html_data = new_html_data \
                .replace("Predicted Label", "Prediction Score")\
                .replace("True Label", f"Token {token_index} ({self.input_tokens[token_index + 1]})")
        elif class_index is not None:
            new_html_data = new_html_data.replace("True Label", self.labels[class_index])
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
            token_class_index_tuples: Optional[Union[List[Tuple[int, int]], None]] = None,
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

        self.input_token_ids = self.tokenizer.encode(text)
        self.input_tokens = [tok.replace("Ġ", "") for tok in self.tokenizer.convert_ids_to_tokens(self.input_token_ids)]
        self.input_length = len(self.input_tokens)
        self.attributions = []
        self.pred_probs = [[] for _ in range(self.input_length)]
        self.labels = list(self.id2label.values())  # assumes that it is sorted
        self.label_probs_dict = {}
        explainer = None

        for label_j in range(self.model.config.num_labels):
            self.attributions.append([])
            self.pred_probs.append([])
            self.label_probs_dict[self.id2label[label_j]] = []

            for token_i in range(self.input_length):
                if self.input_token_ids[token_i] in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]\
                        or (class_index is not None and label_j != class_index)\
                        or (token_index is not None and token_i != token_index) \
                        or (token_class_index_tuples is not None and (token_i, label_j) not in token_class_index_tuples):
                    self.attributions[label_j].append(None)
                    self.pred_probs[token_i].append(None)
                    self.label_probs_dict[self.id2label[label_j]].append(None)
                else:
                    print(f'Attribution for class={label_j} token={token_i}', end='\r', flush=True)
                    explainer = SequenceClassificationExplainer(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        attribution_type=self.attribution_type,
                        token_index=[token_i],
                    )
                    explainer(text, label_j, None, embedding_type)

                    self.attributions[label_j].append(explainer.attributions)
                    self.pred_probs[token_i].append(explainer.pred_probs)
                    self.label_probs_dict[self.id2label[label_j]].append(explainer.pred_probs)

        self.input_ids = explainer.input_ids if explainer is not None else None
        return self.word_attributions

    def __str__(self):
        s = f"{self.__class__.__name__}("
        s += f"\n\tmodel={self.model.__class__.__name__},"
        s += f"\n\ttokenizer={self.tokenizer.__class__.__name__},"
        s += f"\n\tattribution_type='{self.attribution_type}',"
        s += f"\n\tcustom_labels={self.custom_labels},"
        s += ")"

        return s
