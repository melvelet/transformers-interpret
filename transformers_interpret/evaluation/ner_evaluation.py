from typing import List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, Pipeline

from transformers_interpret import TokenClassificationExplainer


def get_topk_rationale(attributions, k: int, return_mask: bool = False):
    if len(attributions) == 0:
        return []
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
        self.id2label = self.model.config.id2label
        self.entities = None
        self.explainer = None
        self.input_str = None
        self.input_tokens = None
        self.input_token_ids = None

    def execute_base_classification(self):
        self.entities = self.pipeline(self.input_str)

    def calculate_attribution_scores(self):
        self.explainer = TokenClassificationExplainer(self.model, self.tokenizer, self.attribution_type)
        token_class_index_tuples = [(e['index'], self.label2id[e['entity']]) for e in self.entities]
        self.explainer(self.input_str, token_class_index_tuples=token_class_index_tuples)
        self.input_token_ids = self.explainer.input_token_ids
        self.input_tokens = self.explainer.input_tokens
        word_attributions = self.explainer.word_attributions
        for e in self.entities:
            e['attribution_scores'] = word_attributions[e['entity']][e['index']]
            e['comprehensiveness'] = dict()
            e['sufficiency'] = dict()

    def calculate_comprehensiveness(self, k):
        for e in self.entities:
            rationale = get_topk_rationale(e['attribution_scores'], k)
            masked_input = torch.tensor([self.input_token_ids])
            for i in rationale:
                masked_input[0][i] = self.tokenizer.mask_token_id
            pred = self.model(masked_input)
            scores = torch.softmax(pred.logits, dim=-1)[0]
            new_conf = scores[e['index']][self.label2id[e['entity']]].item()
            # new_label = self.id2label[scores[e['index']].argmax(axis=-1).item()]
            # print('old_conf', e['score'], 'new_conf', new_conf, 'old_label', e['entity'], 'new_label', new_label, 'diff', e['score'] - new_conf)
            e['comprehensiveness'][str(k)] = e['score'] - new_conf

    def calculate_sufficiency(self, k):
        for e in self.entities:
            rationale = get_topk_rationale(e['attribution_scores'], k)
            masked_input = torch.tensor([self.input_token_ids])
            for i, _ in enumerate(masked_input[0][1:-1]):
                if i + 1 not in rationale:
                    masked_input[0][i + 1] = self.tokenizer.mask_token_id
            pred = self.model(masked_input)
            scores = torch.softmax(pred.logits, dim=-1)[0]
            new_conf = scores[e['index']][self.label2id[e['entity']]].item()
            # new_label = self.id2label[scores[e['index']].argmax(axis=-1).item()]
            # print('old_conf', e['score'], 'new_conf', new_conf, 'old_label', e['entity'], 'new_label', new_label, 'diff', e['score'] - new_conf)
            e['sufficiency'][str(k)] = e['score'] - new_conf

    def __call__(self, input: str, k_values: List[int] = [1]):
        self.input_str = input
        self.execute_base_classification()
        self.calculate_attribution_scores()
        for k in k_values:
            self.calculate_comprehensiveness(k)
            self.calculate_sufficiency(k)
        # print(self.entities)
