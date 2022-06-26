from datetime import datetime
from statistics import mean, median, stdev, variance, StatisticsError
from typing import List, Dict
import torch
from datasets import Dataset
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


def calculate_compsuff(comprehensiveness: Dict, sufficiency: Dict):
    return {k: comprehensiveness[k] - sufficiency[k]
            for k in comprehensiveness}


class NERSentenceEvaluator:
    def __init__(self,
                 pipeline: Pipeline,
                 attribution_type: str = "lig"):
        self.pipeline = pipeline
        self.model: PreTrainedModel = pipeline.model
        self.tokenizer: PreTrainedTokenizer = pipeline.tokenizer
        self.attribution_type: str = attribution_type
        self.label2id = self.model.config.label2id
        self.id2label = self.model.config.id2label
        self.entities = None
        self.explainer = TokenClassificationExplainer(self.model, self.tokenizer, self.attribution_type)
        self.input_str = None
        self.input_tokens = None
        self.input_token_ids = None

    def execute_base_classification(self):
        self.entities = self.pipeline(self.input_str)

    def calculate_attribution_scores(self):
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
            e['comprehensiveness'][k] = e['score'] - new_conf

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
            e['sufficiency'][k] = e['score'] - new_conf

    def calculate_average_scores_for_sentence(self):
        def _calculate_mean(attr: str, squared: bool = False):
            entities = self.entities[0][attr] if self.entities else []
            if squared:
                return {k: mean([e[attr][k]**2 for e in self.entities])
                        for k in entities}
            return {k: mean([e[attr][k] for e in self.entities])
                    for k in entities}

        if self.entities is None \
                or (len(self.entities) > 0
                    and (len(self.entities[0]['comprehensiveness']) == 0
                    or len(self.entities[0]['sufficiency']) == 0)):
            raise ValueError("Scores have not yet been calculated. Please call the evaluator on text first.")

        if len(self.entities) == 0:
            return None

        return {
            'mean': {
                'comprehensiveness': _calculate_mean('comprehensiveness'),
                'sufficiency': _calculate_mean('sufficiency'),
                'compdiff': calculate_compsuff(
                    comprehensiveness=_calculate_mean('comprehensiveness'),
                    sufficiency=_calculate_mean('sufficiency'),
                )
            },
            'squared_mean': {
                'comprehensiveness': _calculate_mean('comprehensiveness', squared=True),
                'sufficiency': _calculate_mean('sufficiency', squared=True),
                'compdiff': calculate_compsuff(
                    comprehensiveness=_calculate_mean('comprehensiveness', squared=True),
                    sufficiency=_calculate_mean('sufficiency', squared=True),
                )
            },
        }

    def __call__(self, input_: str, k_values: List[int] = [1]):
        self.input_str = input_
        self.execute_base_classification()
        self.calculate_attribution_scores()
        for k in k_values:
            self.calculate_comprehensiveness(k)
            self.calculate_sufficiency(k)

        return {
            'scores': self.calculate_average_scores_for_sentence(),
            'entities': self.entities,
            'tokens': len(self.input_tokens),
        }


class NERDatasetEvaluator:
    def __init__(self,
                 pipeline: Pipeline,
                 dataset: Dataset,
                 attribution_type: str = "lig",
                 ):
        self.pipeline = pipeline
        self.dataset = dataset if isinstance(dataset, list) else [dataset]
        self.attribution_type = attribution_type
        self.evaluator = NERSentenceEvaluator(self.pipeline, self.attribution_type)
        self.raw_scores: List[Dict] = []
        self.raw_entities: List[Dict] = []
        self.scores = None

    def calculate_average_scores_for_dataset(self, k_values):
        def _calculate_statistical_function(attr: str, squared: bool = False, func: str = None):
            if func == 'median':
                func = median
            elif func == 'stdev':
                func = stdev
            elif func == 'variance':
                func = variance
            else:
                func = mean

            try:
                if squared:
                    return {k: func([e['squared_mean'][attr][k]**2 for e in self.raw_scores if e is not None])
                            for k in k_values}
                return {k: func([e['mean'][attr][k] for e in self.raw_scores if e is not None])
                        for k in k_values}
            except StatisticsError:
                print(f"Can't calculate {func.__name__}. Too few data points...")
                return None

        if len(self.raw_scores) == 0:
            raise ValueError("Scores have not yet been calculated. Please call the evaluator on a dataset first.")

        # Check if at least one sentence has a score
        if not any(self.raw_scores):
            return None

        return {
            'mean': {
                'comprehensiveness': _calculate_statistical_function('comprehensiveness'),
                'sufficiency': _calculate_statistical_function('sufficiency'),
                'compdiff': calculate_compsuff(
                    comprehensiveness=_calculate_statistical_function('comprehensiveness'),
                    sufficiency=_calculate_statistical_function('sufficiency'),
                )
            },
            'squared_mean': {
                'comprehensiveness': _calculate_statistical_function('comprehensiveness', squared=True),
                'sufficiency': _calculate_statistical_function('sufficiency', squared=True),
                'compdiff': calculate_compsuff(
                    comprehensiveness=_calculate_statistical_function('comprehensiveness', squared=True),
                    sufficiency=_calculate_statistical_function('sufficiency', squared=True),
                )
            },
            'median': {
                'comprehensiveness': _calculate_statistical_function('comprehensiveness', func='median'),
                'sufficiency': _calculate_statistical_function('sufficiency', func='median'),
            },
            'stdev': {
                'comprehensiveness': _calculate_statistical_function('comprehensiveness', func='stdev'),
                'sufficiency': _calculate_statistical_function('sufficiency', func='stdev'),
            },
            'variance': {
                'comprehensiveness': _calculate_statistical_function('comprehensiveness', func='variance'),
                'sufficiency': _calculate_statistical_function('sufficiency', func='variance'),
            },
        }

    def __call__(self, k_values: List[int] = [1]):
        passages = 0
        entities = 0
        tokens = 0
        passages_without_entities = 0
        start_time = datetime.now()
        i = 0
        for split in self.dataset:
            for document in split:
                i += 1
                if i > 5:
                    break
                for passage in document['passages']:
                    passages += 1
                    print('Passage', passages, end='\r', flush=True)
                    if len(passage['text']) > 1:
                        print('len(passage[\'text\']) > 1', passage)
                        exit(-1)
                    result = self.evaluator(passage['text'][0], k_values)
                    self.raw_scores.append(result['scores'])
                    self.raw_entities.append(result['entities'])
                    entities += len(result['entities'])
                    if len(result['entities']) == 0:
                        passages_without_entities += 1
                    tokens += result['tokens']

        end_time = datetime.now()
        duration = end_time - start_time
        self.scores = {
            'scores': self.calculate_average_scores_for_dataset(k_values),
            'stats': {
                'passages': passages,
                'entities': entities,
                'avg_entities': entities / passages,
                'tokens': tokens,
                'avg_tokens': tokens / passages,
                'passages_without_entities': passages_without_entities,
            },
            'settings': {
                'model': self.pipeline.model.config._name_or_path,
                'tokenizer': self.pipeline.tokenizer.name_or_path,
                'dataset': [dataset.info.config_name for dataset in self.dataset],
                'attribution_type': self.attribution_type,
                'k_values': k_values,
            },
            'timing': {
                'start_time': str(start_time),
                'end_time': str(end_time),
                'duration': str(duration),
                'per_passage': str(duration / passages),
                'per_entity': str(duration / entities),
                'per_token': str(duration / tokens),
            },
        }

        return self.scores

