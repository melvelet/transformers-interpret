from datetime import datetime
from statistics import mean, median, stdev, variance, StatisticsError
from typing import List, Dict, Union, Optional
import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer, Pipeline

from transformers_interpret.evaluation import InputPreProcessor
from transformers_interpret import TokenClassificationExplainer


def get_rationale(attributions, k: int, continuous: bool = False, return_mask: bool = False):
    if continuous:
        return get_continuous_rationale(attributions, k, return_mask)
    return get_topk_rationale(attributions, k, return_mask)


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


def get_continuous_rationale(attributions, k: int, return_mask: bool = False):
    if len(attributions) == 0 or return_mask:
        return []

    tensor = torch.FloatTensor([a[1] for a in attributions])
    scores: List[float] = list()
    for i, _ in enumerate(tensor[:len(tensor) - k + 1]):
        scores.append(sum(tensor[i:i+k]))

    argmax = torch.argmax(torch.FloatTensor(scores)).item()
    indices = [i for i in range(argmax, argmax + k)]

    return indices


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
            e['rationales'] = dict()

    def calculate_comprehensiveness(self, k: int, continuous: bool = False):
        for e in self.entities:
            rationale = get_rationale(e['attribution_scores'], k, continuous)
            masked_input = torch.tensor([self.input_token_ids])
            for i in rationale:
                masked_input[0][i] = self.tokenizer.mask_token_id
            pred = self.model(masked_input)
            scores = torch.softmax(pred.logits, dim=-1)[0]
            new_conf = scores[e['index']][self.label2id[e['entity']]].item()
            e['comprehensiveness'][k] = e['score'] - new_conf

    def calculate_sufficiency(self, k: int, continuous: bool = False):
        for e in self.entities:
            rationale = get_rationale(e['attribution_scores'], k, continuous)
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

    def write_rationales(self, k: int, continuous: bool = False):
        for e in self.entities:
            e['rationales'][k] = get_rationale(e['attribution_scores'], k, continuous)

    def get_all_scores_in_sentence(self, k_values):
        document_scores = list()
        for e in self.entities:
            entity_scores = {
                'comprehensiveness': {k: e['comprehensiveness'][k] for k in k_values},
                'sufficiency': {k: e['sufficiency'][k] for k in k_values},
                'compdiff': {k: e['comprehensiveness'][k] - e['sufficiency'][k] for k in k_values},
            }
            document_scores.append(entity_scores)

        return document_scores

    def __call__(self, input_: str, k_values: List[int] = [1], continuous: bool = False):
        self.input_str = input_
        self.execute_base_classification()
        self.calculate_attribution_scores()
        for k in k_values:
            self.write_rationales(k, continuous=continuous)
            self.calculate_comprehensiveness(k, continuous=continuous)
            self.calculate_sufficiency(k, continuous=continuous)

        return {
            'scores': self.get_all_scores_in_sentence(k_values),
            'entities': self.entities,
            'tokens': len(self.input_tokens),
        }


class NERDatasetEvaluator:
    def __init__(self,
                 pipeline: Pipeline,
                 dataset: Union[Dataset, List[Dataset]],
                 attribution_type: str = "lig",
                 ):
        self.pipeline = pipeline
        self.input_pre_processor = InputPreProcessor(self.pipeline.tokenizer, max_tokens=512)
        self.dataset = dataset
        self.label2id = self.input_pre_processor.get_labels_from_dataset(self.dataset)
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
                    return {k: func([e[attr][k]**2 for e in self.raw_scores])
                            for k in k_values}
                return {k: func([e[attr][k] for e in self.raw_scores])
                        for k in k_values}
            except StatisticsError:
                print(f"Can't calculate {func.__name__}. Too few data points...")
                return None

        if len(self.raw_scores) == 0:
            raise ValueError("Scores have not yet been calculated. Please call the evaluator on a dataset first.")

        return {
            'mean': {
                'comprehensiveness': _calculate_statistical_function('comprehensiveness'),
                'sufficiency': _calculate_statistical_function('sufficiency'),
                'compdiff': _calculate_statistical_function('compdiff'),
            },
            'squared_mean': {
                'comprehensiveness': _calculate_statistical_function('comprehensiveness', squared=True),
                'sufficiency': _calculate_statistical_function('sufficiency', squared=True),
                'compdiff': _calculate_statistical_function('compdiff', squared=True),
            },
            'median': {
                'comprehensiveness': _calculate_statistical_function('comprehensiveness', func='median'),
                'sufficiency': _calculate_statistical_function('sufficiency', func='median'),
                'compdiff': _calculate_statistical_function('compdiff', func='median'),
            },
            'stdev': {
                'comprehensiveness': _calculate_statistical_function('comprehensiveness', func='stdev'),
                'sufficiency': _calculate_statistical_function('sufficiency', func='stdev'),
                'compdiff': _calculate_statistical_function('compdiff', func='stdev'),
            },
            'variance': {
                'comprehensiveness': _calculate_statistical_function('comprehensiveness', func='variance'),
                'sufficiency': _calculate_statistical_function('sufficiency', func='variance'),
                'compdiff': _calculate_statistical_function('compdiff', func='variance'),
            },
        }

    def __call__(self, k_values: List[int] = [1], continuous: bool = False, max_documents: Optional[Union[int, None]] = None):
        passages = 0
        entities = 0
        tokens = 0
        passages_without_entities = 0
        truncated_tokens = 0
        truncated_documents = 0
        start_time = datetime.now()
        for split in self.dataset:
            for document in self.dataset[split]:
                if max_documents and passages > max_documents:
                    break
                passages += 1
                print('Passage', passages)
                if max([len(i['text']) for i in document['passages']]) > 1:
                    print('len(passage[\'text\']) > 1', document)
                    exit(-1)
                input_document = self.input_pre_processor(document)
                truncated_tokens += self.input_pre_processor.stats['truncated_tokens']
                truncated_documents += 1 if self.input_pre_processor.stats['is_truncated'] > 0 else 0
                result = self.evaluator(input_document, k_values, continuous)
                self.raw_scores.extend(result['scores'])
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
                'splits': len(self.dataset),
                'passages': passages,
                'entities': entities,
                'avg_entities': entities / passages,
                'tokens': tokens,
                'avg_tokens': tokens / passages,
                'passages_without_entities': passages_without_entities,
                'truncated_documents': truncated_documents,
                'truncated_documents_ratio': truncated_documents / passages,
                'truncated_tokens': truncated_tokens,
                'avg_truncated_tokens': truncated_tokens / truncated_documents if truncated_documents > 0 else 0,
            },
            'settings': {
                'model': self.pipeline.model.config._name_or_path,
                'tokenizer': self.pipeline.tokenizer.name_or_path,
                'dataset': list(set([dataset.info.config_name for dataset in self.dataset])),
                'splits': [split for split in self.dataset[0].info.splits],
                'attribution_type': self.attribution_type,
                'k_values': k_values,
                'continuous': continuous,
            },
            'timing': {
                'start_time': str(start_time),
                'end_time': str(end_time),
                'duration': str(duration),
                'per_k_value': str(duration / len(k_values)),
                'per_passage': str(duration / passages),
                'per_entity': str(duration / entities),
                'per_token': str(duration / tokens),
            },
        }

        return self.scores

