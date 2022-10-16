import math
import os
from datetime import datetime
from statistics import mean, median, stdev, variance, StatisticsError
from typing import List, Dict, Union, Optional

import numpy as np
import torch
from alive_progress import alive_bar
from datasets import Dataset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, Pipeline
from transformers_interpret.evaluation.input_pre_processor import get_labels_from_dataset
from transformers_interpret import TokenClassificationExplainer

CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES') if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
CUDA_DEVICE = torch.device('cpu') if CUDA_VISIBLE_DEVICES == 'cpu' else torch.device('cuda')
if os.environ.get('BATCH_SIZE'):
    BATCH_SIZE = os.environ.get('BATCH_SIZE')
else:
    BATCH_SIZE = 16 if CUDA_VISIBLE_DEVICES != 'cpu' else 32


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
    if len(attributions) >= k:
        indices = torch.topk(tensor, k).indices
    else:
        indices = torch.topk(tensor, len(attributions)).indices

    if return_mask:
        mask = [0 for _ in range(len(attributions))]
        for i in indices:
            if i not in [0, len(attributions) - 1]:
                mask[i] = 1
        return mask

    # print(tensor[indices[0]], tensor.shape, indices.tolist())
    return indices.tolist()


def get_continuous_rationale(attributions, k: int, return_mask: bool = False):
    if len(attributions) == 0 or return_mask:
        return []

    tensor = torch.FloatTensor([a[1] for a in attributions])
    scores: List[float] = list()
    if k >= len(attributions):
        k = len(attributions)
    for i, _ in enumerate(tensor[:len(tensor) - k + 1]):
        scores.append(sum(tensor[i:i + k]))

    argmax = torch.argmax(torch.FloatTensor(scores)).item()
    indices = [i for i in range(argmax, argmax + k)]

    return indices


class NERSentenceAttributor:
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
        pred = self.model(torch.tensor([self.input_token_ids]).to(CUDA_DEVICE))
        scores = torch.softmax(pred.logits, dim=-1)[0]
        if self.relevant_class_names is not None:
            for i, gold_label in enumerate(self.gold_labels):
                entity = [e for e in self.entities if e['index'] == i]
                doc_len = len(self.input_document['passages'][0]['text'][0])
                if entity:
                    entity = entity[0]
                    pred_label = self.label2id[entity['entity']]
                    entity['gold_label'] = gold_label
                    entity['pred_label'] = pred_label
                    entity['doc_id'] = self.input_document['id']
                    entity['doc_doc_id'] = self.input_document['document_id']
                    entity['doc_title'] = self.input_document['passages'][0]['text'][0][
                                          :20 if doc_len >= 20 else doc_len]  # was self.input_document['passages'][1]['text'][0]
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
                        entity['other_score'] = np.float64(scores[i][gold_label].item())
                        entity['other_entity'] = self.id2label[gold_label]
                    else:
                        entity['other_score'] = 0.0
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
                        'doc_title': self.input_document['passages'][0]['text'][0][:20 if doc_len >= 20 else doc_len],
                        'score': np.float64(scores[i][gold_label].item()),
                    }
                    # print('created', gold_label, self.label2id['O'], entity['eval'])
                    self.entities.append(entity)

            entities_before = len(self.entities)
            self.entities = list(filter(lambda x: x['eval'] != 'TN', self.entities))
            self.discarded_entities = entities_before - len(self.entities)
        test = [e for e in self.entities if 'eval' not in e]
        assert len(test) == 0, f"{len(test)} out of {len(self.entities)} entities without eval: {test}"
        # print('self.entities if eval not in e', test)

    def calculate_attribution_scores(self):
        print(
            f"calculate_attribution_scores for {len(self.entities) + len([e['other_entity'] for e in self.entities if e['other_entity'] is not None])} entities")
        token_class_index_tuples = [(e['index'], self.label2id[e['entity']]) for e in self.entities]
        token_class_index_tuples += [(e['index'], self.label2id[e['other_entity']]) for e in self.entities if
                                     e['other_entity'] is not None]
        self.explainer(self.input_str, token_class_index_tuples=token_class_index_tuples,
                       internal_batch_size=BATCH_SIZE)
        # print('assert', [t for t in self.input_token_ids if t != 1], self.explainer.input_token_ids)
        self.input_token_ids = self.explainer.input_token_ids
        self.input_tokens = self.explainer.input_tokens
        word_attributions = self.explainer.word_attributions
        for e in self.entities:
            for prefix in self.prefixes:
                if prefix == 'other_' and e['other_entity'] in [None, 'O']:
                    continue
                e[f'{prefix}attribution_scores'] = word_attributions[e[f'{prefix}entity']][e['index']]
                # print('prefix', prefix, 'class', e[f'{prefix}entity'], 'index', e['index'], e[f'{prefix}attribution_scores'])
                if len(self.input_token_ids) != len(e[f'{prefix}attribution_scores']):
                    raise Exception(
                        f"attribution_scores of length {len(e['attribution_scores'])} while input tokens have length of {len(self.input_token_ids)}")
                # print('attribution_scores length, input:', len(self.input_token_ids), 'attribution_scores:',  len(e['attribution_scores']))
                e[f'{prefix}comprehensiveness'] = {'top_k': dict(), 'continuous': dict(), 'bottom_k': dict()}
                e[f'{prefix}sufficiency'] = {'top_k': dict(), 'continuous': dict(), 'bottom_k': dict(),
                                             'other_top_k': dict(), 'other_continuous': dict(),
                                             'other_bottom_k': dict()}
                e[f'{prefix}rationales'] = {'top_k': dict(), 'continuous': dict(), 'bottom_k': dict(),
                                            'other_top_k': dict(), 'other_continuous': dict(), 'other_bottom_k': dict()}
            # bar()

    def __call__(self, input_document, evaluate_other: bool = False):
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

        return {
            'entities': self.entities,
            'discarded_entities': self.discarded_entities,
            'tokens': len(self.input_token_ids),
        }


class NERDatasetAttributor:
    def __init__(self,
                 pipeline: Pipeline,
                 dataset: Dataset,
                 attribution_type: str = "lig",
                 class_name: str = None,
                 ):
        self.pipeline = pipeline
        self.pipeline.model.to(CUDA_DEVICE)
        self.dataset = dataset
        self.label2id, self.id2label = get_labels_from_dataset(dataset, has_splits=False)
        print('label2id', self.label2id)
        self.attribution_type = attribution_type
        self.attributor = NERSentenceAttributor(self.pipeline, self.attribution_type, class_name=class_name,
                                                dataset_name=self.dataset.info.config_name)
        self.entities: List[Dict] = []
        self.class_name = class_name
        self.relevant_class_names = [f"B-{class_name}", f"I-{class_name}"] if class_name else None
        self.relevant_class_indices = [self.label2id[c] for c in self.relevant_class_names] if class_name else None
        self.attributed_entities = 0

    def __call__(self,
                 max_documents: Optional[Union[int, None]] = None,
                 start_document: Optional[Union[int, None]] = None,
                 evaluate_other: bool = False,
                 ):
        documents = 0
        discarded_entities = 0
        tokens = 0
        start_time = datetime.now()

        with alive_bar(total=len(self.dataset)) as bar:
            for document in self.dataset:
                torch.cuda.empty_cache()
                documents += 1
                if start_document and documents < start_document:
                    continue
                if max_documents and 0 < max_documents < documents:
                    break
                print('Document', documents)
                result = self.attributor(document,
                                         evaluate_other=evaluate_other)
                self.entities.append({
                    'entities': result['entities'],
                    'document_id': document['document_id'],
                    'discarded_entities': result['discarded_entities'],
                })
                discarded_entities += result['discarded_entities']
                self.attributed_entities += len([e for e in result['entities']])
                self.attributed_entities += len(
                    [e['other_entity'] for e in result['entities'] if e['other_entity'] is not None])
                tokens += result['tokens']
                bar()

        end_time = datetime.now()
        duration = end_time - start_time
        found_entities = len([e for doc in self.entities for e in doc['entities']])
        self.stats = {
            'stats': {
                'total_documents': len(self.dataset),
                'processed_documents': documents,
                'discarded_entities': discarded_entities,
            },
            'settings': {
                'model': self.pipeline.model.config._name_or_path,
                'tokenizer': self.pipeline.tokenizer.name_or_path,
                'dataset': self.dataset.info.config_name,
                'attribution_type': self.attribution_type,
                'start_document': start_document,
                'max_documents': max_documents,
                'evaluate_other': evaluate_other,
                'class_name': self.class_name,
            },
            'timing': {
                'start_time': str(start_time),
                'end_time': str(end_time),
                'duration': str(duration),
                'per_document': str(duration / documents),
                'per_entity': str(duration / found_entities),
                'per_attributed_entity': str(duration / max(1, self.attributed_entities)),
                'per_token': str(duration / tokens),
            },
        }

        return self.stats


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
        self.input_str = None
        self.input_tokens = None
        self.input_token_ids = None
        self.discarded_entities = 0
        self.dataset_name = dataset_name

    def calculate_measure(self, k_values: List[int], measure, continuous: bool = False, bottom_k: bool = False):
        # print('calculate_comprehensiveness, k=', k, 'continuous=', continuous, 'bottom_k=', bottom_k)
        masked_inputs = torch.full(
            size=(len(self.entities) * len(self.prefixes) * len(k_values), len(self.input_token_ids)),
            fill_value=self.tokenizer.pad_token_id,
            dtype=torch.int32,
        )

        i = -1
        for k in k_values:
            for e in self.entities:
                for prefix in self.prefixes:
                    i += 1
                    if prefix == 'other_' and e['other_entity'] in [None, 'O']:
                        continue

                    masked_inputs[i] = torch.tensor(self.input_token_ids)

                    if measure == 'comprehensiveness':
                        rationale = get_rationale(e[f'{prefix}attribution_scores'], k, continuous,
                                                  bottom_k=bottom_k if not continuous else False)
                        # print(rationale, len(self.input_token_ids), 'bottom_k', bottom_k, 'continuous', continuous)
                        for j in rationale:
                            masked_inputs[i][j] = self.tokenizer.mask_token_id
                    elif measure == 'sufficiency':
                        rationale = get_rationale(e[f'{prefix}attribution_scores'], k, continuous,
                                                  bottom_k=bottom_k if not continuous else False)
                        for j, _ in enumerate(masked_inputs[i][1:-1]):
                            if j + 1 not in rationale:
                                masked_inputs[i][j + 1] = self.tokenizer.mask_token_id

        preds = []
        with torch.no_grad():
            for i in range(math.ceil(masked_inputs.shape[0] / BATCH_SIZE)):
                # print('batch', i, end='\r', flush=True)
                batch = masked_inputs[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :].to(CUDA_DEVICE)
                preds.append(self.model(batch).logits)
                # torch.cuda.empty_cache()
            if preds:
                preds = torch.cat(preds, dim=0)
            # print(preds.shape)

        i = -1
        for k in k_values:
            for e in self.entities:
                for prefix in self.prefixes:
                    i += 1
                    if prefix == 'other_' and e['other_entity'] in [None, 'O']:
                        continue
                    scores = torch.softmax(preds, dim=-1)[i]
                    new_conf = scores[e['index']][self.label2id[e[f'{prefix}entity']]].item()
                    mode = 'top_k'
                    if continuous:
                        mode = 'continuous'
                    elif bottom_k:
                        mode = 'bottom_k'
                    # if mode == 'bottom_k':
                    #     print(e[f'{prefix}score'], '-', new_conf, '=', e[f'{prefix}score'] - new_conf)
                    #     print(scores)
                    e[f'{prefix}{measure}'][mode][k] = e[f'{prefix}score'] - new_conf

    def write_rationales(self, k: int, continuous: bool = False, bottom_k: bool = False):
        for e in self.entities:
            for prefix in self.prefixes:
                if prefix == 'other_' and e['other_entity'] in [None, 'O']:
                    continue
                e['rationales'][f'{prefix}top_k'][k] = get_rationale(e[f'{prefix}attribution_scores'], k,
                                                                     continuous=False)
                if continuous:
                    e['rationales'][f'{prefix}continuous'][k] = get_rationale(e[f'{prefix}attribution_scores'], k,
                                                                              continuous=True)
                if bottom_k:
                    e['rationales'][f'{prefix}bottom_k'][k] = get_rationale(e[f'{prefix}attribution_scores'], k,
                                                                            continuous=False, bottom_k=True)

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
                        'other_compdiff': {
                            mode: {k: e['other_comprehensiveness'][mode][k] - e['other_sufficiency'][mode][k] for k in
                                   k_values}
                            for mode in modes},
                    }
                )

            document_scores.append(entity_scores)

        return document_scores

    def __call__(self, input_document, attributions, k_values: List[int] = [1], continuous: bool = False,
                 bottom_k: bool = False,
                 evaluate_other: bool = False):
        self.input_document = input_document
        print('document_id', self.input_document['document_id'])
        self.prefixes = ['']
        if evaluate_other:
            self.prefixes.append('other_')
        self.input_str = input_document['text']
        self.input_token_ids = input_document['input_ids']
        self.gold_labels = input_document['labels']
        self.entities = attributions['entities']
        self.discarded_entities = attributions['discarded_entities']
        modes = ['top_k']
        for k in k_values:
            self.write_rationales(k, continuous=continuous, bottom_k=bottom_k)

        print('calculate top_k scores')
        self.calculate_measure(k_values, 'comprehensiveness')
        self.calculate_measure(k_values, 'sufficiency')

        if continuous:
            print('calculate continuous scores')
            modes.append('continuous')
            self.calculate_measure(k_values, 'comprehensiveness', continuous=True)
            self.calculate_measure(k_values, 'sufficiency', continuous=True)

        if bottom_k:
            print('calculate bottom_k scores')
            modes.append('bottom_k')
            self.calculate_measure(k_values, 'comprehensiveness', bottom_k=True)
            self.calculate_measure(k_values, 'sufficiency', bottom_k=True)

        # print('collect scores')

        return {
            'scores': self.get_all_scores_in_sentence(k_values, modes),
            'entities': self.entities,
            'discarded_entities': self.discarded_entities,
            'tokens': len(self.input_token_ids),
        }


class NERDatasetEvaluator:
    def __init__(self,
                 pipeline: Pipeline,
                 dataset: Dataset,
                 attributions,
                 attribution_type: str = "lig",
                 class_name: str = None,
                 ):
        self.pipeline = pipeline
        self.pipeline.model.to(CUDA_DEVICE)
        self.dataset = dataset
        self.label2id, self.id2label = get_labels_from_dataset(dataset, has_splits=False)
        print('label2id', self.label2id)
        self.attribution_type = attribution_type
        self.attributions = attributions
        self.evaluator = NERSentenceEvaluator(self.pipeline, self.attribution_type, class_name=class_name,
                                              dataset_name=self.dataset.info.config_name)
        self.tokenizer = self.pipeline.tokenizer
        self.raw_scores: List[Dict] = []
        self.raw_entities: List[Dict] = []
        self.scores = None
        self.class_name = class_name
        self.relevant_class_names = [f"B-{class_name}", f"I-{class_name}"] if class_name else None
        self.relevant_class_indices = [self.label2id[c] for c in self.relevant_class_names] if class_name else None

    def calculate_average_scores_for_dataset(self, k_values, modes):
        def _calculate_statistical_function(attr: str, func: str = None, eval_=None, take_best_rationale: bool = False,
                                            take_best_rationale_threshold: float = 0.05):
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
                if take_best_rationale:
                    mode = 'top_k'
                    best_rationale_scores = []
                    best_rationale_k_values = []
                    for e in self.raw_scores:
                        best_rationale_compdiff = -1
                        best_rationale_per_mode_k_value = 0
                        best_rationale_compdiff_prev = 0
                        prev_k = k_values[-1]
                        for k in reversed(k_values):
                            if best_rationale_compdiff_prev - e['compdiff'][mode][k] > take_best_rationale_threshold:
                                best_rationale_compdiff = e[attr][mode][k]
                                best_rationale_per_mode_k_value = prev_k
                                break
                            best_rationale_compdiff_prev = e['compdiff'][mode][k]
                            prev_k = k
                        if best_rationale_compdiff == -1:
                            best_rationale_compdiff = e[attr][mode][2]
                            best_rationale_per_mode_k_value = 2
                        best_rationale_scores.append(best_rationale_compdiff)
                        best_rationale_k_values.append(best_rationale_per_mode_k_value)
                    return func(best_rationale_scores), func(best_rationale_k_values)

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
                'comprehensiveness_Best_5': _calculate_statistical_function('comprehensiveness',
                                                                            take_best_rationale=True,
                                                                            take_best_rationale_threshold=0.05),
                'sufficiency_Best_5': _calculate_statistical_function('sufficiency', take_best_rationale=True,
                                                                      take_best_rationale_threshold=0.05),
                'compdiff_Best_5': _calculate_statistical_function('compdiff', take_best_rationale=True,
                                                                   take_best_rationale_threshold=0.05),
                'comprehensiveness_Best_10': _calculate_statistical_function('comprehensiveness',
                                                                             take_best_rationale=True,
                                                                             take_best_rationale_threshold=0.1),
                'sufficiency_Best_10': _calculate_statistical_function('sufficiency', take_best_rationale=True,
                                                                       take_best_rationale_threshold=0.1),
                'compdiff_Best_10': _calculate_statistical_function('compdiff', take_best_rationale=True,
                                                                    take_best_rationale_threshold=0.1),
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
                'comprehensiveness_FN_Switched': _calculate_statistical_function('comprehensiveness',
                                                                                 eval_=['FN', 'Switched']),
                'sufficiency_FN_Switched': _calculate_statistical_function('sufficiency', eval_=['FN', 'Switched']),
                'compdiff_FN_Switched': _calculate_statistical_function('compdiff', eval_=['FN', 'Switched']),
                'comprehensiveness_Switched': _calculate_statistical_function('comprehensiveness', eval_=['Switched']),
                'sufficiency_Switched': _calculate_statistical_function('sufficiency', eval_=['Switched']),
                'compdiff_Switched': _calculate_statistical_function('compdiff', eval_=['Switched']),
                'comprehensiveness_Error': _calculate_statistical_function('comprehensiveness',
                                                                           eval_=['FN', 'FP', 'Switched']),
                'sufficiency_Error': _calculate_statistical_function('sufficiency', eval_=['FN', 'FP', 'Switched']),
                'compdiff_Error': _calculate_statistical_function('compdiff', eval_=['FN', 'FP', 'Switched']),
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
                'other_comprehensiveness_Switched': _calculate_statistical_function('other_comprehensiveness',
                                                                                    eval_=['Switched']),
                'other_sufficiency_Switched': _calculate_statistical_function('other_sufficiency', eval_=['Switched']),
                'other_compdiff_Switched': _calculate_statistical_function('other_compdiff', eval_=['Switched']),
                'other_comprehensiveness_FN_Switched': _calculate_statistical_function('other_comprehensiveness',
                                                                                       eval_=['FN', 'Switched']),
                'other_sufficiency_FN_Switched': _calculate_statistical_function('other_sufficiency',
                                                                                 eval_=['FN', 'Switched']),
                'other_compdiff_FN_Switched': _calculate_statistical_function('other_compdiff',
                                                                              eval_=['FN', 'Switched']),
                'other_comprehensiveness_Error': _calculate_statistical_function('other_comprehensiveness',
                                                                                 eval_=['FN', 'FP', 'Switched']),
                'other_sufficiency_Error': _calculate_statistical_function('other_sufficiency',
                                                                           eval_=['FN', 'FP', 'Switched']),
                'other_compdiff_Error': _calculate_statistical_function('other_compdiff',
                                                                        eval_=['FN', 'FP', 'Switched']),
            },
            'median': {
                'comprehensiveness_Best_5': _calculate_statistical_function('comprehensiveness',
                                                                            func='median',
                                                                            take_best_rationale=True,
                                                                            take_best_rationale_threshold=0.05),
                'sufficiency_Best_5': _calculate_statistical_function('sufficiency',
                                                                      func='median',
                                                                      take_best_rationale=True,
                                                                      take_best_rationale_threshold=0.05),
                'compdiff_Best_5': _calculate_statistical_function('compdiff',
                                                                   func='median',
                                                                   take_best_rationale=True,
                                                                   take_best_rationale_threshold=0.05),
                'comprehensiveness_Best_10': _calculate_statistical_function('comprehensiveness',
                                                                             func='median',
                                                                             take_best_rationale=True,
                                                                             take_best_rationale_threshold=0.1),
                'sufficiency_Best_10': _calculate_statistical_function('sufficiency',
                                                                       func='median',
                                                                       take_best_rationale=True,
                                                                       take_best_rationale_threshold=0.1),
                'compdiff_Best_10': _calculate_statistical_function('compdiff',
                                                                    func='median', take_best_rationale=True,
                                                                    take_best_rationale_threshold=0.1),
                'comprehensiveness': _calculate_statistical_function('comprehensiveness', func='median'),
                'sufficiency': _calculate_statistical_function('sufficiency', func='median'),
                'compdiff': _calculate_statistical_function('compdiff', func='median'),
                'other_comprehensiveness': _calculate_statistical_function('other_comprehensiveness', func='median'),
                'other_sufficiency': _calculate_statistical_function('other_sufficiency', func='median'),
                'other_compdiff': _calculate_statistical_function('other_compdiff', func='median'),
            },
            'stdev': {
                'comprehensiveness_Best_5': _calculate_statistical_function('comprehensiveness', func='stdev',
                                                                            take_best_rationale=True,
                                                                            take_best_rationale_threshold=0.05),
                'sufficiency_Best_5': _calculate_statistical_function('sufficiency', func='stdev',
                                                                      take_best_rationale=True,
                                                                      take_best_rationale_threshold=0.05),
                'comprehensiveness_Best_10': _calculate_statistical_function('comprehensiveness', func='stdev',
                                                                             take_best_rationale=True,
                                                                             take_best_rationale_threshold=0.1),
                'sufficiency_Best_10': _calculate_statistical_function('sufficiency', func='stdev',
                                                                       take_best_rationale=True,
                                                                       take_best_rationale_threshold=0.1),
                'compdiff_Best': _calculate_statistical_function('compdiff', func='stdev', take_best_rationale=True),
                'comprehensiveness': _calculate_statistical_function('comprehensiveness', func='stdev'),
                'sufficiency': _calculate_statistical_function('sufficiency', func='stdev'),
                'compdiff': _calculate_statistical_function('compdiff', func='stdev'),
                'other_comprehensiveness': _calculate_statistical_function('other_comprehensiveness', func='stdev'),
                'other_sufficiency': _calculate_statistical_function('other_sufficiency', func='stdev'),
                'other_compdiff': _calculate_statistical_function('other_compdiff', func='stdev'),
            },
            'variance': {
                'comprehensiveness_Best_5': _calculate_statistical_function('comprehensiveness', func='variance',
                                                                            take_best_rationale=True,
                                                                            take_best_rationale_threshold=0.05),
                'sufficiency_Best_5': _calculate_statistical_function('sufficiency', func='variance',
                                                                      take_best_rationale=True,
                                                                      take_best_rationale_threshold=0.05),
                'comprehensiveness_Best_10': _calculate_statistical_function('comprehensiveness', func='variance',
                                                                             take_best_rationale=True,
                                                                             take_best_rationale_threshold=0.1),
                'sufficiency_Best_10': _calculate_statistical_function('sufficiency', func='variance',
                                                                       take_best_rationale=True,
                                                                       take_best_rationale_threshold=0.1),
                'compdiff_Best': _calculate_statistical_function('compdiff', func='variance', take_best_rationale=True),
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
                 exclude_reference_token: bool = False,
                 ):
        documents = 0
        found_entities = 0
        attributed_entities = 0
        annotated_entities = 0
        annotated_entities_positive = 0
        discarded_entities = 0
        tokens = 0
        documents_without_entities = 0
        first_entity_no_word = 0
        start_time = datetime.now()

        self.ordered_attributions: List[Dict] = []
        for doc in self.dataset:
            attr = [e for e in self.attributions if e['document_id'] == doc['document_id']]
            assert len(attr) == 1
            self.ordered_attributions.append(attr[0])

        with alive_bar(total=len(self.ordered_attributions) - start_document) as bar:
            for document, doc_attributions in tqdm(zip(self.dataset, self.ordered_attributions)):
                if len(doc_attributions['entities']) > 0:
                    first_entity = doc_attributions['entities'][0]
                    # print(first_entity)
                    if first_entity['index'] < len(document['input_ids']):
                        if 'word' in first_entity:
                            entity_word = first_entity['word'].replace('Ġ', '').strip()
                            doc_word = self.tokenizer.decode(document['input_ids'][first_entity['index']]).replace('Ġ', '').strip()
                            assert entity_word == doc_word, f"{entity_word} ({len(entity_word)}) != {doc_word} ({len(doc_word)})"
                        else:
                            first_entity_no_word += 1
                    if len(first_entity['attribution_scores']) != len(document['input_ids']):
                        print('truncate', len(first_entity['attribution_scores']), 'to', len(document['input_ids']))
                        doc_attributions['entities'] = list(
                            filter(lambda x: x['index'] <= len(document['input_ids']) - 1,
                                   doc_attributions['entities']))
                        for e in doc_attributions['entities']:
                            e['attribution_scores'] = e['attribution_scores'][:len(document['input_ids']) - 1]
                            if 'other_attribution_scores' in e:
                                e['other_attribution_scores'] = e['other_attribution_scores'][
                                                                :len(document['input_ids']) - 1]

                assert document['document_id'] == doc_attributions[
                    'document_id'], f"{document['document_id']} --- {doc_attributions['document_id']}"
                documents += 1
                if start_document and documents < start_document:
                    continue
                if max_documents and 0 < max_documents < documents:
                    break
                print('Document', documents)
                result = self.evaluator(document,
                                        attributions=doc_attributions,
                                        k_values=k_values,
                                        continuous=continuous,
                                        bottom_k=bottom_k,
                                        evaluate_other=evaluate_other)
                # print('Save scores')
                self.raw_scores.extend(result['scores'])
                self.raw_entities.extend(result['entities'])
                discarded_entities += result['discarded_entities']
                annotated_entities += len([label for label in document['labels'] if label != 0])
                annotated_entities_positive += len(
                    [label for label in document['labels'] if label in self.relevant_class_indices])
                # found_entities += len(result['entities'])
                # attributed_entities_raw = [e['other_entity'] for e in result['entities'] if e['other_entity'] is not None]
                # print('attributed_entities_raw', len(attributed_entities_raw), attributed_entities_raw)
                # test_other = [e['other_entity'] for e in result['entities']]
                # test_eval = [e['eval'] for e in result['entities']]
                # print('test_other', test_other)
                # print('test_eval', test_eval)
                attributed_entities += len([e for e in result['entities']])
                attributed_entities += len(
                    [e['other_entity'] for e in result['entities'] if e['other_entity'] is not None])
                if len(result['entities']) == 0:
                    documents_without_entities += 1
                tokens += result['tokens']
                bar()

        end_time = datetime.now()
        duration = end_time - start_time
        modes = ['top_k']
        found_entities = len([e for e in self.raw_entities if e['eval'] in ['TP', 'Switched', 'FP', 'FN']])
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
                'avg_annotated_entities_all_classes': annotated_entities / max(1, documents),
                'avg_annotated_entities': annotated_entities_positive / max(1, documents),
                'found_entities': found_entities,
                'found_entities_tp': len([e for e in self.raw_entities if e['eval'] == 'TP']),
                'found_entities_fp': len([e for e in self.raw_entities if e['eval'] == 'FP']),
                'found_entities_fn': len([e for e in self.raw_entities if e['eval'] == 'FN']),
                'found_entities_switched': len([e for e in self.raw_entities if e['eval'] == 'Switched']),
                'found_entities_all_classes': found_entities + discarded_entities,
                'attributed_entities': attributed_entities,
                'discarded_entities': discarded_entities,
                'avg_found_entities_per_document': found_entities / max(1, documents),
                'avg_found_entities_per_document_all_classes': (found_entities + discarded_entities) / max(1, documents),
                'found_to_annotated_entities_ratio': found_entities / max(1, annotated_entities_positive),
                'found_to_annotated_entities_ratio_all_classes': (
                                                                         found_entities + discarded_entities) / max(1, annotated_entities),
                'tokens': tokens,
                'avg_tokens_per_document': tokens / documents,
                'documents_without_entities': documents_without_entities,
                'first_entity_no_word': first_entity_no_word,
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
                'class_name': self.class_name,
                'exclude_reference_token': exclude_reference_token,
            },
            'timing': {
                'start_time': str(start_time),
                'end_time': str(end_time),
                'duration': str(duration),
                'per_k_value': str(duration / max(1, len(k_values))),
                'per_document': str(duration / max(1, documents)),
                'per_entity': str(duration / max(1, found_entities)),
                'per_attributed_entity': str(duration / max(1, attributed_entities)),
                'per_token': str(duration / tokens),
            },
        }

        return self.scores
