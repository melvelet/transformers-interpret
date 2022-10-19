import json
import math
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
from bigbio.dataloader import BigBioConfigHelpers
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline

from transformers_interpret import TokenClassificationExplainer
from transformers_interpret.evaluation.input_pre_processor import get_labels_from_dataset, InputPreProcessor

continuous = True
bottom_k = True
evaluate_other = True

attribution_types = [
    'lig',
    'lgxa',
    'lfa',
    'gradcam',
]

dataset_names = [
    'bc5cdr_bigbio_kb',
    'euadr_bigbio_kb',
    'ncbi_disease_bigbio_kb',
    'scai_disease_bigbio_kb',
    'ddi_corpus_bigbio_kb',
    'mlee_bigbio_kb',
    'cadec_bigbio_kb',
]

huggingface_models = [
    'biolinkbert',
    'bioelectra-discriminator',
    'roberta',
    'electra',
    'bert',
]

CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES') if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
CUDA_DEVICE = torch.device('cpu') if CUDA_VISIBLE_DEVICES == 'cpu' else torch.device('cuda')
if os.environ.get('BATCH_SIZE'):
    BATCH_SIZE = os.environ.get('BATCH_SIZE')
else:
    BATCH_SIZE = 16 if CUDA_VISIBLE_DEVICES != 'cpu' else 32


def generate_latex_text(attributions,
                        tokens,
                        reference_token_idx,
                        rationale_1=None,
                        rationale_2=None,
                        collapse_threshold=.05,
                        collapse_margin=15,
                        show_idx_of_rationale=False,
                        ):
    if rationale_1 is None:
        rationale_1 = []
    # if rationale_2 is None:
    #     rationale_2 = []
    # rationale_2 = list(filter(lambda x: x not in rationale_1, rationale_2))
    latex_text = ""
    # print(len(attributions), len(tokens))

    # Pre_processing
    tokens = [t.replace('%', '\\%').replace('Ġ', '##') for t in tokens]
    # print(reference_token_idx, rationale_1)
    for idx in rationale_1:
        if idx == reference_token_idx:
            continue
        else:
            if show_idx_of_rationale:
                tokens[idx] = f"{tokens[idx]}(idx: {idx})"
            tokens[idx] = f"\\dotuline{{{tokens[idx]}}}"
            tokens[idx] = f"\\textit{{{tokens[idx]}}}"

    if reference_token_idx >= 0:
        tokens[reference_token_idx] = f"\\underline{{{tokens[reference_token_idx]}}}"
        tokens[reference_token_idx] = f"\\textbf{{{tokens[reference_token_idx]}}}"

    # Auto-collapsing
    indices_below_threshold = [0] * (len(attributions) + 2 * collapse_margin)
    if collapse_threshold > 0 and collapse_margin > 0:
        for i, attr in enumerate(attributions):
            if abs(attr[1]) < collapse_threshold and i not in rationale_1:
                indices_below_threshold[i + collapse_margin] = 1

    attrs = [attr[1] for attr in attributions]
    # print(attrs)
    # print(indices_below_threshold)
    collapsing = False
    for i, data in enumerate(zip(attributions, tokens)):
        attr, token = data
        if token in ['[SEP]', '[CLS]']:
            continue
        if 0 not in indices_below_threshold[i:i + 2 * collapse_margin]:
            if not collapsing:
                latex_text += '[...] '
            collapsing = True
        elif attr[1] < 0:
            latex_text += f"\\negrel{{{abs(round(attr[1] * 100, 0))}}}{{{token}}} "
            collapsing = False
        else:
            latex_text += f"\\posrel{{{abs(round(attr[1] * 100, 0))}}}{{{token}}} "
            collapsing = False

    latex_text = latex_text.replace("##", "\\texttt{\\#\\#}")
    # print(latex_text)
    return latex_text


class QualitativeVisualizer:
    def __init__(self):
        self.entities = {}

    def load_dataset(self, dataset=0):
        self.dataset_name = dataset_names[dataset]
        conhelps = BigBioConfigHelpers()
        self.dataset = conhelps.for_config_name(self.dataset_name).load_dataset()
        self.label2id, self.id2label = get_labels_from_dataset(self.dataset)
        if len(self.dataset) > 1:
            self.dataset = self.dataset["test"] if 'test' in self.dataset else self.dataset["validation"]
        else:
            shuffled_dataset = self.dataset["train"].shuffle(seed=42)
            dataset_length = len(shuffled_dataset)
            self.dataset = shuffled_dataset.select(
                range(math.floor(dataset_length * 0.8), math.floor(dataset_length * 0.9)))

    def load_other_pipeline(self, base_path):
        # print(os.getcwd())
        finetuned_huggingface_model = f"{base_path}{self.huggingface_models[1]}/{self.dataset_name.replace('_bigbio_kb', '')}/final"
        other_model: AutoModelForTokenClassification = AutoModelForTokenClassification\
            .from_pretrained(finetuned_huggingface_model,
                             local_files_only=True,
                             num_labels=len(self.label2id)).to(CUDA_DEVICE)
        self.pipeline = TokenClassificationPipeline(model=other_model, tokenizer=self.tokenizers[self.huggingface_models[1]])

    def load_tokenizers(self, models=[1, 2]):
        self.huggingface_models = [huggingface_models[i] for i in models]
        self.tokenizers = {
            'bioelectra-discriminator': AutoTokenizer.from_pretrained(
                'kamalkraj/bioelectra-base-discriminator-pubmed-pmc'),
            'roberta': AutoTokenizer.from_pretrained('Jean-Baptiste/roberta-large-ner-english'),
        }
        self.pre_processors = {
            'bioelectra-discriminator': InputPreProcessor(self.tokenizers['bioelectra-discriminator'],
                                                          [self.tokenizers['roberta']],
                                                          self.label2id,
                                                          max_tokens=512),
            'roberta': InputPreProcessor(self.tokenizers['roberta'],
                                         [self.tokenizers['bioelectra-discriminator']],
                                         self.label2id,
                                         max_tokens=512),
        }

    def load_entities(self, base_path, entity_type=0, attributions=None):
        if attributions is None:
            attributions = [0, 1, 3]
        self.attribution_types = [attribution_types[i] for i in attributions]
        self.entity_type = 'drug' if entity_type == 1 else 'disease'
        exclude_string = 'include'
        for model in self.huggingface_models:
            for attribution_type in self.attribution_types:
                base_file_name = f"{base_path}{self.dataset_name.replace('_bigbio_kb', '')}_{self.entity_type}_{model}_{attribution_type}_{exclude_string}"
                with open(f'{base_file_name}_raw_entities.json', 'r') as f:
                    entities = json.load(f)
                    print(model, attribution_type, len(entities))
                    if model not in self.entities:
                        self.entities[model] = {}
                    self.entities[model][attribution_type] = entities

    def prepare(self, doc_id, ref_token_idx):
        self.doc_id = doc_id
        self.ref_token_idx = ref_token_idx
        self.entity = None
        # for e in self.entities[self.huggingface_models[0]][attribution_types[0]]:
        #     print(e['doc_id'] == str(doc_id), e['doc_id'], str(doc_id), e['index'] == ref_token_idx, e['index'], ref_token_idx)
        #     if e['doc_id'] == str(doc_id) and e['index'] == ref_token_idx:
        #         self.entity = e
        # assert self.entity is not None
        self.entity = [e for e in self.entities[self.huggingface_models[0]][attribution_types[0]]
                       if e['doc_id'] == str(doc_id) and e['index'] == ref_token_idx][0]
        doc = [doc for doc in self.dataset if doc['document_id'] == doc_id][0]
        self.docs = {
            'bioelectra-discriminator': self.pre_processors['bioelectra-discriminator'](doc),
            'roberta': self.pre_processors['roberta'](doc)
        }

    def print_table(self, k_value=5):
        latex_tables = ''
        for model in self.huggingface_models:
            tokens = self.tokenizers[model].batch_decode(self.docs[model]['input_ids'])
            for attribution_type in self.attribution_types:
                entity = [e for e in self.entities[model][attribution_type]
                          if e['doc_id'] == self.doc_id and e['index'] == self.ref_token_idx][0]
                for prefix in ['', 'other_']:
                    text = generate_latex_text(
                        entity[f'{prefix}attribution_scores'],
                        tokens,
                        reference_token_idx=entity['index'],
                        rationale_1=entity['rationales']['top_k'][str(k_value)],
                    )
                    print(f'\n\n{text}')

    def pick_entities(self, eval_=None, doc_id=None, n_value=1, k_values=[5, 10]):
        if eval_:
            for attribution_type in self.attribution_types:
                test = [i for i in self.entities[self.huggingface_models[0]]]
                print(test)
                filtered_entities = list(
                    filter(lambda x: x['eval'] == eval_ and x['entity'].startswith('B'),
                           self.entities[self.huggingface_models[0]][attribution_type]))
                self.entities[self.huggingface_models[0]][attribution_type] = filtered_entities

        if doc_id:
            self.entity = [e for e in self.entities[self.huggingface_models[0]][attribution_types[0]] if e['doc_id'] == str(doc_id)][0]
        else:
            indices = [i for i in range(len(self.entities[self.huggingface_models[0]][attribution_types[0]]))]
            chosen_entities = []
            for n in range(n_value):
                i = indices.pop(random.choice(indices))
                chosen_entities.append(self.entities[self.huggingface_models[0]][attribution_types[0]][i])
            self.entity = chosen_entities[0]

        # print(self.entity)
        doc_id = self.entity['doc_id']
        idx = self.entity['index']
        doc = [doc for doc in self.dataset if doc['document_id'] == doc_id][0]
        self.docs = {
            'bioelectra-discriminator': self.pre_processors['bioelectra-discriminator'](doc),
            'roberta': self.pre_processors['roberta'](doc)
        }
        # print('doc_id', doc_id, docs['bioelectra-discriminator'])
        model = self.huggingface_models[0]
        tokens = self.tokenizers[model].batch_decode(self.docs[model]['input_ids'])
        print(len(tokens))
        text = ''
        for i, tok in enumerate(tokens):
            if tok in ['<s>', '</s>', '<pad>', '[CLS]', '[SEP]']:
                continue
            elif i == idx:
                text += f"{tok}[REF: {i}] "
            elif i in self.entity['rationales']['top_k'][str(k_values[0])]:
                text += f"{tok}[rat: {i}] "
            else:
                text += f"{tok} "

        # latex_texts += f"{text}\n\n"
        print(text)
        return idx, tokens[idx], doc_id

    def find_in_other_model(self, model_1_token, reference_token_idx=-1):
        other_model = self.huggingface_models[1]
        other_doc = self.docs[other_model]
        tokens_other_model = self.tokenizers[other_model].batch_decode(other_doc['input_ids'])
        potential_tokens = []
        for i, tok in enumerate(tokens_other_model):
            if model_1_token.lower() in tok.lower() or tok.lower() in model_1_token.lower():
                potential_tokens.append(i)

        text = ''
        for i, tok in enumerate(tokens_other_model):
            if tok in ['<s>', '</s>', '<pad>']:
                continue
            elif i == reference_token_idx:
                text += f"{tok}[REF: {i}] "
            elif i in potential_tokens:
                text += f"{tok}[tok: {i}] "
            else:
                text += f"{tok} "

        # text = generate_latex_text(
        #     [('', 0)] * len(tokens_other_model),
        #     tokens_other_model,
        #     reference_token_idx=reference_token_idx,
        #     rationale_1=potential_tokens,
        #     collapse_threshold=0,
        #     show_idx_of_rationale=True,
        # )
        print(text)

    def ensure_attr_scores_in_other_model(self, reference_token_idx):
        other_model = self.huggingface_models[1]
        other_doc = self.docs[other_model]
        for attr_type in self.attribution_types:
            self.other_entity = [e for e in self.entities[other_model][attr_type] if
                                 e['doc_id'] == other_doc['document_id'] and e['index'] == reference_token_idx]
            if self.other_entity:
                self.other_entity = self.other_entity[0]
                print('Entity existed already:')
                # print(self.other_entity)
                if not self.other_entity['other_entity']:
                    main_entity_labels = [self.entity['gold_label'], self.entity['pred_label']]
                    main_other_labels = [self.other_entity['gold_label'], self.other_entity['pred_label']]
                    labels_to_attribute = [label for label in main_entity_labels if label not in main_other_labels]
                    if len(labels_to_attribute) == 0 or len(labels_to_attribute) > 1:
                        print('possible error!', main_entity_labels, main_other_labels)
                    else:
                        print(f'get attributions for other entity ({attr_type})')
                        explainer = TokenClassificationExplainer(self.pipeline.model, self.pipeline.tokenizer, attr_type)
                        token_class_index_tuples = [(reference_token_idx, labels_to_attribute[0])]
                        explainer(other_doc['text'], token_class_index_tuples=token_class_index_tuples,
                                  internal_batch_size=BATCH_SIZE)
                        word_attributions = explainer.word_attributions
                        self.other_entity['other_entity'] = self.id2label[labels_to_attribute[0]]
                        self.other_entity['other_attribution_scores'] = word_attributions[self.id2label[labels_to_attribute[0]]][reference_token_idx]

            else:
                print(f'get attributions {attr_type}')
                token_class_index_tuples = [(reference_token_idx, self.entity['gold_label']), (reference_token_idx, self.entity['pred_label'])]
                explainer = TokenClassificationExplainer(self.pipeline.model, self.pipeline.tokenizer, attr_type)
                explainer(other_doc['text'], token_class_index_tuples=token_class_index_tuples,
                          internal_batch_size=BATCH_SIZE)
                word_attributions = explainer.word_attributions
                other_entity = {
                    'eval': 'TN',
                    'index': reference_token_idx,
                    'entity': self.id2label[self.entity['gold_label']],
                    'attribution_scores': word_attributions[self.id2label[self.entity['gold_label']]][reference_token_idx],
                    'other_entity': self.id2label[self.entity['pred_label']],
                    'other_attribution_scores': word_attributions[self.id2label[self.entity['pred_label']]][reference_token_idx],
                }
                self.entities[other_model][attr_type].append(other_entity)

