from pprint import pprint

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, Pipeline

from transformers_interpret import TokenClassificationExplainer


def get_topk_rationale(attributions, k: int, return_mask: bool = False):
    tensor = torch.FloatTensor([a[1] for a in attributions])
    indices = torch.topk(tensor, k).indices

    if return_mask:
        mask = [0 for _ in range(len(attributions))]
        for i in indices:
            mask[i] = 1
        return mask

    return indices.tolist()


class NERSentenceEvaluator:
    def __init__(self,
                 pipeline: Pipeline,
                 attribution_type: str = "lig"):
        self.pipeline = pipeline
        self.model = pipeline.model
        self.tokenizer = pipeline.tokenizer
        self.attribution_type = attribution_type
        self.label2id = self.model.config.label2id
        self.entities = None
        self.explainer = None
        self.input_str = None
        self.input_tokens = None

    def execute_base_classification(self):
        self.entities = self.pipeline(self.input_str)

    def calculate_attribution_scores(self):
        self.explainer = TokenClassificationExplainer(self.model, self.tokenizer, self.attribution_type)
        token_class_index_tuples = [(e['index'], self.label2id[e['entity']]) for e in self.entities]
        self.explainer(self.input_str, token_class_index_tuples=token_class_index_tuples)
        self.input_tokens = self.explainer.input_tokens
        word_attributions = self.explainer.word_attributions
        for e in self.entities:
            e['test'] = word_attributions[e['entity']][e['index']]

    def __call__(self, input):
        self.input_str = input
        self.execute_base_classification()
        self.calculate_attribution_scores()



