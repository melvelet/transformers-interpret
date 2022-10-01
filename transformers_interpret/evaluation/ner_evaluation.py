from datetime import datetime
from statistics import mean, median, stdev, variance, StatisticsError
from typing import List, Dict, Union, Optional

import numpy as np
import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer, Pipeline
from transformers_interpret.evaluation.input_pre_processor import get_labels_from_dataset
from transformers_interpret import TokenClassificationExplainer


def get_rationale(attributions, k: int, continuous: bool = False, return_mask: bool = False, bottom_k: bool = False):
    if continuous:
        return get_continuous_rationale(attributions, k, return_mask)
    return get_topk_rationale(attributions, k, return_mask, bottom_k)


def get_topk_rationale(attributions, k: int, return_mask: bool = False, bottom_k: bool = False):
    if len(attributions) == 0:
        return []
    tensor = torch.FloatTensor([a[1] for a in attributions])
    if bottom_k:
        tensor = - tensor
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
                 attribution_type: str = "lig",
                 class_name: str = None,
                 dataset_name: str = ''):
        self.pipeline = pipeline
        self.model: PreTrainedModel = pipeline.model
        self.tokenizer: PreTrainedTokenizer = pipeline.tokenizer
        self.attribution_type: str = attribution_type
        self.label2id = self.model.config.label2id
        self.id2label = self.model.config.id2label
        self.relevant_class_names = [f"B-{class_name}", f"I-{class_name}"] if class_name else None
        self.relevant_class_indices = [self.label2id[c] for c in self.relevant_class_names] if class_name else None
        self.entities = None
        self.explainer = TokenClassificationExplainer(self.model, self.tokenizer, self.attribution_type)
        self.input_str = None
        self.input_tokens = None
        self.input_token_ids = None
        self.discarded_entities = 0
        self.dataset_name = dataset_name

    def execute_base_classification(self):
        # print('Base classification')
        self.entities = self.pipeline(self.input_str)
        pred = self.model(torch.tensor([self.input_token_ids]))
        scores = torch.softmax(pred.logits, dim=-1)[0]
        if self.relevant_class_names is not None:
            for i, gold_label in enumerate(self.gold_labels):
                entity = [e for e in self.entities if e['index'] == i]
                if entity:
                    entity = entity[0]
                    pred_label = self.label2id[entity['entity']]
                    entity['gold_label'] = gold_label
                    entity['pred_label'] = pred_label
                    entity['doc_id'] = self.input_document['id']
                    if self.dataset_name.startswith('euadr'):
                        entity['doc_title'] = self.input_document['passages'][1]['text'][0]
                    if gold_label in self.relevant_class_indices:
                        if gold_label == pred_label:
                            entity['eval'] = 'TP'
                        elif pred_label in self.relevant_class_indices:
                            entity['eval'] = 'Switched'
                        else:
                            entity['eval'] = 'FN'
                    else:
                        if pred_label in self.relevant_class_indices:
                            entity['eval'] = 'FP'
                        else:
                            entity['eval'] = 'TN'
                    # print(gold_label, pred_label, entity['eval'])

                    if entity['eval'] in ['FN', 'FP', 'Switched']:
                        entity['entity'] = self.id2label[gold_label]
                        entity['other_score'] = entity['score']
                        entity['score'] = np.float64(scores[i][gold_label].item())
                        entity['other_entity'] = self.id2label[gold_label]
                    else:
                        entity['other_entity'] = None
                elif gold_label in self.relevant_class_indices:
                    entity = {
                        'entity': self.id2label[gold_label],
                        'other_score': 0.0,
                        'index': i,
                        'gold_label': gold_label,
                        'pred_label': self.label2id['O'],
                        'other_entity': None,
                        'eval': 'FN',
                        'doc_id': self.input_document['id'],
                        'score': np.float64(scores[i][gold_label].item()),
                    }
                    if self.dataset_name.startswith('euadr'):
                        entity['doc_title'] = self.input_document['passages'][1]['text'][0]
                    # print('created', gold_label, self.label2id['O'], entity['eval'])
                    self.entities.append(entity)

            entities_before = len(self.entities)
            self.entities = list(filter(lambda x: x['eval'] != 'TN', self.entities))
            self.discarded_entities = entities_before - len(self.entities)
        test = [e for e in self.entities if 'eval' not in e]
        assert len(test) == 0
        # print('self.entities if eval not in e', test)

    def calculate_attribution_scores(self):
        print(f'calculate_attribution_scores for {len(self.entities)} entities')
        token_class_index_tuples = [(e['index'], self.label2id[e['entity']]) for e in self.entities]
        token_class_index_tuples += [(e['index'], e['other_entity']) for e in self.entities if e['other_entity'] is not None]
        self.explainer(self.input_str, token_class_index_tuples=token_class_index_tuples)
        # print('assert', [t for t in self.input_token_ids if t != 1], self.explainer.input_token_ids)
        self.input_token_ids = self.explainer.input_token_ids
        self.input_tokens = self.explainer.input_tokens
        word_attributions = self.explainer.word_attributions
        for e in self.entities:
            for prefix in self.prefixes:
                if prefix == 'other_' and e['other_entity'] in [None, 'O']:
                    continue
                e[f'{prefix}attribution_scores'] = word_attributions[e[f'{prefix}entity']][e['index']]
                if len(self.input_token_ids) != len(e[f'{prefix}attribution_scores']):
                    raise Exception(f"attribution_scores of length {len(e['attribution_scores'])} while input tokens have length of {len(self.input_token_ids)}")
                # print('attribution_scores length, input:', len(self.input_token_ids), 'attribution_scores:',  len(e['attribution_scores']))
                e[f'{prefix}comprehensiveness'] = {'top_k': dict(), 'continuous': dict(), 'bottom_k': dict()}
                e[f'{prefix}sufficiency'] = {'top_k': dict(), 'continuous': dict(), 'bottom_k': dict(),
                                             'other_top_k': dict(), 'other_continuous': dict(), 'other_bottom_k': dict()}
                e[f'{prefix}rationales'] = {'top_k': dict(), 'continuous': dict(), 'bottom_k': dict(),
                                            'other_top_k': dict(), 'other_continuous': dict(), 'other_bottom_k': dict()}

    def calculate_comprehensiveness(self, k: int, continuous: bool = False, bottom_k: bool = False):
        print('calculate_comprehensiveness, k=', k, 'continuous=', continuous, 'bottom_k=', bottom_k)
        for e in self.entities:
            for prefix in self.prefixes:
                if prefix == 'other_' and e['other_entity'] in [None, 'O']:
                    continue
                rationale = get_rationale(e[f'{prefix}attribution_scores'], k, continuous, bottom_k=bottom_k if not continuous else False)
                masked_input = torch.tensor([self.input_token_ids])
                for i in rationale:
                    masked_input[0][i] = self.tokenizer.mask_token_id
                pred = self.model(masked_input)
                scores = torch.softmax(pred.logits, dim=-1)[0]
                new_conf = scores[e['index']][self.label2id[e[f'{prefix}entity']]].item()
                mode = 'top_k'
                if continuous:
                    mode = 'continuous'
                elif bottom_k:
                    mode = 'bottom_k'
                e[f'{prefix}comprehensiveness'][mode][k] = e[f'{prefix}score'] - new_conf

    def calculate_sufficiency(self, k: int, continuous: bool = False, bottom_k: bool = False):
        print('calculate_sufficiency, k=', k, 'continuous=', continuous, 'bottom_k=', bottom_k)
        for e in self.entities:
            for prefix in self.prefixes:
                if prefix == 'other_' and e['other_entity'] in [None, 'O']:
                    continue
                rationale = get_rationale(e[f'{prefix}attribution_scores'], k, continuous, bottom_k=bottom_k if not continuous else False)
                masked_input = torch.tensor([self.input_token_ids])
                for i, _ in enumerate(masked_input[0][1:-1]):
                    if i + 1 not in rationale:
                        masked_input[0][i + 1] = self.tokenizer.mask_token_id
                pred = self.model(masked_input)
                scores = torch.softmax(pred.logits, dim=-1)[0]
                new_conf = scores[e['index']][self.label2id[e[f'{prefix}entity']]].item()
                # new_label = self.id2label[scores[e['index']].argmax(axis=-1).item()]
                # print('old_conf', e['score'], 'new_conf', new_conf, 'old_label', e['entity'], 'new_label', new_label, 'diff', e['score'] - new_conf)
                mode = 'top_k'
                if continuous:
                    mode = 'continuous'
                elif bottom_k:
                    mode = 'bottom_k'
                e[f'{prefix}sufficiency'][mode][k] = e[f'{prefix}score'] - new_conf

    def write_rationales(self, k: int, continuous: bool = False, bottom_k: bool = False):
        for e in self.entities:
            for prefix in self.prefixes:
                if prefix == 'other_' and e['other_entity'] in [None, 'O']:
                    continue
                e['rationales'][f'{prefix}top_k'][k] = get_rationale(e[f'{prefix}attribution_scores'], k, continuous=False)
                if continuous:
                    e['rationales'][f'{prefix}continuous'][k] = get_rationale(e[f'{prefix}attribution_scores'], k, continuous=True)
                if bottom_k:
                    e['rationales'][f'{prefix}bottom_k'][k] = get_rationale(e[f'{prefix}attribution_scores'], k, continuous=False, bottom_k=True)

    def get_all_scores_in_sentence(self, k_values, modes):
        document_scores = list()
        for e in self.entities:
            entity_scores = {
                'eval': e['eval'],
                'comprehensiveness': {mode: {k: e['comprehensiveness'][mode][k] for k in k_values}
                                      for mode in modes},
                'sufficiency': {mode: {k: e['sufficiency'][mode][k] for k in k_values}
                                for mode in modes},
                'compdiff': {mode: {k: e['comprehensiveness'][mode][k] - e['sufficiency'][mode][k] for k in k_values}
                             for mode in modes},
            }
            if e['other_entity'] not in [None, 'O']:
                entity_scores.update(
                    {
                        'other_comprehensiveness': {mode: {k: e['other_comprehensiveness'][mode][k] for k in k_values}
                                                    for mode in modes},
                        'other_sufficiency': {mode: {k: e['other_sufficiency'][mode][k] for k in k_values}
                                              for mode in modes},
                        'other_compdiff': {mode: {k: e['other_comprehensiveness'][mode][k] - e['other_sufficiency'][mode][k] for k in k_values}
                                           for mode in modes},
                    }
                )

            document_scores.append(entity_scores)

        return document_scores

    def __call__(self, input_document, k_values: List[int] = [1], continuous: bool = False, bottom_k: bool = False, evaluate_other: bool = False):
        self.input_document = input_document
        print('document_id', self.input_document['document_id'])
        self.prefixes = ['']
        if evaluate_other:
            self.prefixes.append('other_')
        self.input_str = input_document['text']
        self.input_token_ids = input_document['input_ids']
        self.gold_labels = input_document['labels']
        self.execute_base_classification()
        self.calculate_attribution_scores()
        modes = ['top_k']
        for k in k_values:
            self.write_rationales(k, continuous=continuous, bottom_k=bottom_k)

        for k in k_values:
            self.calculate_comprehensiveness(k)
            self.calculate_sufficiency(k)

        if continuous:
            modes.append('continuous')
            for k in k_values:
                self.calculate_comprehensiveness(k, continuous=True)
                self.calculate_sufficiency(k, continuous=True)

        if bottom_k:
            modes.append('bottom_k')
            for k in k_values:
                self.calculate_comprehensiveness(k, bottom_k=True)
                self.calculate_sufficiency(k, bottom_k=True)

        print('collect scores')

        return {
            'scores': self.get_all_scores_in_sentence(k_values, modes),
            'entities': self.entities,
            'discarded_entities': self.discarded_entities,
            'tokens': len(self.input_tokens),
        }


class NERDatasetEvaluator:
    def __init__(self,
                 pipeline: Pipeline,
                 dataset: Dataset,
                 attribution_type: str = "lig",
                 class_name: str = None,
                 ):
        self.pipeline = pipeline
        self.dataset = dataset
        self.label2id, self.id2label = get_labels_from_dataset(dataset, has_splits=False)
        print('label2id', self.label2id)
        self.attribution_type = attribution_type
        self.evaluator = NERSentenceEvaluator(self.pipeline, self.attribution_type, class_name=class_name, dataset_name=self.dataset.info.config_name)
        self.raw_scores: List[Dict] = []
        self.raw_entities: List[Dict] = []
        self.scores = None
        self.relevant_class_names = [f"B-{class_name}", f"I-{class_name}"] if class_name else None
        self.relevant_class_indices = [self.label2id[c] for c in self.relevant_class_names] if class_name else None

    def calculate_average_scores_for_dataset(self, k_values, modes):
        def _calculate_statistical_function(attr: str, squared: bool = False, func: str = None, eval_=None):
            if eval_ is None:
                eval_ = ['TP', 'FN', 'FP', 'Switched']
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
                    return {mode:
                                {k: func([e[attr][mode][k]**2 for e in self.raw_scores if attr in e and e['eval'] in eval_])
                                    for k in k_values}
                            for mode in modes}

                # test = [e[attr][modes[0]][k_values[0]] for e in self.raw_scores if attr in e and e['eval'] in eval_]
                # test_types = [type(v) for v in test]
                # if len(set(test_types)) > 1:
                #     print('attr', attr, 'func', func, 'eval_', eval_)
                #     print(test)
                #     print(test_types)
                return {mode:
                            {k: func([e[attr][mode][k] for e in self.raw_scores if attr in e and e['eval'] in eval_])
                             for k in k_values}
                        for mode in modes}
            except StatisticsError:
                print(f"Can't calculate {func.__name__} for attribute {attr} ({eval_}). Too few data points...")
                return None

        if len(self.raw_scores) == 0:
            raise ValueError("Scores have not yet been calculated. Please call the evaluator on a dataset first.")

        return {
            'mean': {
                'comprehensiveness': _calculate_statistical_function('comprehensiveness'),
                'sufficiency': _calculate_statistical_function('sufficiency'),
                'compdiff': _calculate_statistical_function('compdiff'),
                'comprehensiveness_TP': _calculate_statistical_function('comprehensiveness', eval_=['TP']),
                'sufficiency_TP': _calculate_statistical_function('sufficiency', eval_=['TP']),
                'compdiff_TP': _calculate_statistical_function('compdiff', eval_=['TP']),
                'comprehensiveness_FP': _calculate_statistical_function('comprehensiveness', eval_=['FP']),
                'sufficiency_FP': _calculate_statistical_function('sufficiency', eval_=['FP']),
                'compdiff_FP': _calculate_statistical_function('compdiff', eval_=['FP']),
                'comprehensiveness_FN': _calculate_statistical_function('comprehensiveness', eval_=['FN']),
                'sufficiency_FN': _calculate_statistical_function('sufficiency', eval_=['FN']),
                'compdiff_FN': _calculate_statistical_function('compdiff', eval_=['FN']),
                'comprehensiveness_Switched': _calculate_statistical_function('comprehensiveness', eval_=['Switched']),
                'sufficiency_Switched': _calculate_statistical_function('sufficiency', eval_=['Switched']),
                'compdiff_Switched': _calculate_statistical_function('compdiff', eval_=['Switched']),
                'other_comprehensiveness': _calculate_statistical_function('other_comprehensiveness'),
                'other_sufficiency': _calculate_statistical_function('other_sufficiency'),
                'other_compdiff': _calculate_statistical_function('other_compdiff'),
                'other_comprehensiveness_TP': _calculate_statistical_function('other_comprehensiveness', eval_=['TP']),
                'other_sufficiency_TP': _calculate_statistical_function('other_sufficiency', eval_=['TP']),
                'other_compdiff_TP': _calculate_statistical_function('other_compdiff', eval_=['TP']),
                'other_comprehensiveness_FP': _calculate_statistical_function('other_comprehensiveness', eval_=['FP']),
                'other_sufficiency_FP': _calculate_statistical_function('other_sufficiency', eval_=['FP']),
                'other_compdiff_FP': _calculate_statistical_function('other_compdiff', eval_=['FP']),
                'other_comprehensiveness_FN': _calculate_statistical_function('other_comprehensiveness', eval_=['FN']),
                'other_sufficiency_FN': _calculate_statistical_function('other_sufficiency', eval_=['FN']),
                'other_compdiff_FN': _calculate_statistical_function('other_compdiff', eval_=['FN']),
                'other_comprehensiveness_Switched': _calculate_statistical_function('other_comprehensiveness', eval_=['Switched']),
                'other_sufficiency_Switched': _calculate_statistical_function('other_sufficiency', eval_=['Switched']),
                'other_compdiff_Switched': _calculate_statistical_function('other_compdiff', eval_=['Switched']),
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
                'other_comprehensiveness': _calculate_statistical_function('other_comprehensiveness', func='median'),
                'other_sufficiency': _calculate_statistical_function('other_sufficiency', func='median'),
                'other_compdiff': _calculate_statistical_function('other_compdiff', func='median'),
            },
            'stdev': {
                'comprehensiveness': _calculate_statistical_function('comprehensiveness', func='stdev'),
                'sufficiency': _calculate_statistical_function('sufficiency', func='stdev'),
                'compdiff': _calculate_statistical_function('compdiff', func='stdev'),
                'other_comprehensiveness': _calculate_statistical_function('other_comprehensiveness', func='stdev'),
                'other_sufficiency': _calculate_statistical_function('other_sufficiency', func='stdev'),
                'other_compdiff': _calculate_statistical_function('other_compdiff', func='stdev'),
            },
            'variance': {
                'comprehensiveness': _calculate_statistical_function('comprehensiveness', func='variance'),
                'sufficiency': _calculate_statistical_function('sufficiency', func='variance'),
                'compdiff': _calculate_statistical_function('compdiff', func='variance'),
                'other_comprehensiveness': _calculate_statistical_function('other_comprehensiveness', func='variance'),
                'other_sufficiency': _calculate_statistical_function('other_sufficiency', func='variance'),
                'other_compdiff': _calculate_statistical_function('other_compdiff', func='variance'),
            },
        }

    def __call__(self,
                 k_values: List[int] = [1],
                 continuous: bool = False,
                 bottom_k: bool = False,
                 max_documents: Optional[Union[int, None]] = None,
                 start_document: Optional[Union[int, None]] = None,
                 evaluate_other: bool = False,
                 ):
        documents = 0
        found_entities = 0
        attributed_entities = 0
        annotated_entities = 0
        annotated_entities_positive = 0
        discarded_entities = 0
        tokens = 0
        documents_without_entities = 0
        start_time = datetime.now()

        for document in self.dataset:
            documents += 1
            if start_document and documents < start_document:
                continue
            if max_documents and 0 < max_documents < documents:
                break
            print('Document', documents)
            print('Evaluate')
            result = self.evaluator(document,
                                    k_values=k_values,
                                    continuous=continuous,
                                    bottom_k=bottom_k,
                                    evaluate_other=evaluate_other)
            # print('Save scores')
            self.raw_scores.extend(result['scores'])
            self.raw_entities.extend(result['entities'])
            discarded_entities += result['discarded_entities']
            annotated_entities += len([label for label in document['labels'] if label != 0])
            annotated_entities_positive += len([label for label in document['labels'] if label in self.relevant_class_indices])
            # found_entities += len(result['entities'])
            attributed_entities_raw = [e['other_entity'] for e in result['entities'] if e['other_entity'] is not None]
            # print('attributed_entities_raw', len(attributed_entities_raw), attributed_entities_raw)
            test_other = [e['other_entity'] for e in result['entities']]
            test_eval = [e['eval'] for e in result['entities']]
            # print('test_other', test_other)
            # print('test_eval', test_eval)
            attributed_entities += len([e['other_entity'] for e in result['entities'] if e['other_entity'] is not None])
            if len(result['entities']) == 0:
                documents_without_entities += 1
            tokens += result['tokens']

        end_time = datetime.now()
        duration = end_time - start_time
        modes = ['top_k']
        found_entities = len([e for e in self.raw_entities if e['eval'] in ['TP', 'Switched', 'FP']])
        if bottom_k:
            modes.append('bottom_k')
        if continuous:
            modes.append('continuous')
        self.scores = {
            'scores': self.calculate_average_scores_for_dataset(k_values, modes),
            'stats': {
                'total_documents': len(self.dataset),
                'processed_documents': documents,
                'annotated_entities_all_classes': annotated_entities,
                'annotated_entities': annotated_entities_positive,
                'avg_annotated_entities_all_classes': annotated_entities / documents,
                'avg_annotated_entities': annotated_entities_positive / documents,
                'found_entities': found_entities,
                'found_entities_tp': len([e for e in self.raw_entities if e['eval'] == 'TP']),
                'found_entities_fp': len([e for e in self.raw_entities if e['eval'] == 'FP']),
                'found_entities_fn': len([e for e in self.raw_entities if e['eval'] == 'FN']),
                'found_entities_switched': len([e for e in self.raw_entities if e['eval'] == 'Switched']),
                'found_entities_all_classes': found_entities + discarded_entities,
                'attributed_entities': attributed_entities,
                'discarded_entities': discarded_entities,
                'avg_found_entities_per_document': found_entities / documents,
                'avg_found_entities_per_document_all_classes': (found_entities + discarded_entities) / documents,
                'found_to_annotated_entities_ratio': found_entities / annotated_entities_positive,
                'found_to_annotated_entities_ratio_all_classes': (found_entities + discarded_entities) / annotated_entities,
                'tokens': tokens,
                'avg_tokens_per_document': tokens / documents,
                'documents_without_entities': documents_without_entities,
            },
            'settings': {
                'model': self.pipeline.model.config._name_or_path,
                'tokenizer': self.pipeline.tokenizer.name_or_path,
                'dataset': self.dataset.info.config_name,
                'attribution_type': self.attribution_type,
                'k_values': k_values,
                'continuous': continuous,
                'bottom_k': bottom_k,
                'start_document': start_document,
                'max_documents': max_documents,
                'evaluate_other': evaluate_other,
            },
            'timing': {
                'start_time': str(start_time),
                'end_time': str(end_time),
                'duration': str(duration),
                'per_k_value': str(duration / len(k_values)),
                'per_document': str(duration / documents),
                'per_entity': str(duration / found_entities),
                'per_attributed_entity': str(duration / max(1, attributed_entities)),
                'per_token': str(duration / tokens),
            },
        }

        return self.scores

