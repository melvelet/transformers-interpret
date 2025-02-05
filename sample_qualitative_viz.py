import json
import math
import os
import random

import torch
from bigbio.dataloader import BigBioConfigHelpers
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline

from transformers_interpret import TokenClassificationExplainer
from transformers_interpret.evaluation import get_rationale
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

CLASS_NAMES = {
    'B-CompositeMention': 'Composite',
    'B-SpecificDisease': 'SpecificDisease',
    'B-DiseaseClass': 'DiseaseClass',
    'I-DiseaseClass': 'DiseaseClass',
    'I-SpecificDisease': 'SpecificDisease',
    'B-Disease': 'Disease',
    'B-Finding': 'Finding',
    'I-Disease': 'Disease',
    'B-Chemical': 'Chemical',
    'I-Chemical': 'Chemical',
    'B-ADR': 'ADR',
    'B-Drug': 'Drug',
    'B-DRUG': 'Drug',
    'B-GROUP': 'Drug',
    'B-BRAND': 'Brand',
    'B-DRUG_N': 'Drug_N',
    'B-Modifier': 'Modifier',
    'O': 'O',
}


def generate_latex_text(attributions,
                        tokens,
                        reference_token_idx,
                        rationale_1=None,
                        rationale_2=None,
                        collapse_threshold=.05,
                        collapse_margin=15,
                        is_roberta=False,
                        ):
    if rationale_1 is None:
        rationale_1 = []
    # if rationale_2 is None:
    #     rationale_2 = []
    # rationale_2 = list(filter(lambda x: x not in rationale_1, rationale_2))
    latex_text = ""
    # print(len(attributions), len(tokens))

    # Pre_processing
    tokens = [t.replace('%', '\\%').replace('-', '--') for t in tokens]
    if is_roberta:
        for i, tok in enumerate(tokens):
            if tok.startswith('Ġ'):
                tokens[i] = tok.replace('Ġ', '')
            elif tok == 'Ċ':
                tokens[i] = '\n'
            elif tok in [',', ':', '.', '-', '(', ')', '<s>', '</s>']:
                continue
            else:
                tokens[i] = f"##{tok}"
    # print(reference_token_idx, rationale_1)
    for idx in rationale_1:
        if idx >= len(tokens):
            continue
        tokens[idx] = f"\\textit{{{tokens[idx]}}}"
        if idx == reference_token_idx:
            continue
        else:
            tokens[idx] = f"\\dotuline{{{tokens[idx]}}}"

    if reference_token_idx >= 0:
        tokens[reference_token_idx] = f"\\underline{{{tokens[reference_token_idx]}}}"
        tokens[reference_token_idx] = f"\\textbf{{{tokens[reference_token_idx]}}}"

    # Auto-collapsing
    indices_below_threshold = [0] * (len(attributions) + 2 * collapse_margin)
    if collapse_threshold > 0 and collapse_margin > 0:
        for i, attr in enumerate(attributions):
            if abs(attr[1]) < collapse_threshold and i not in rationale_1:
                indices_below_threshold[i + collapse_margin] = 1

    # attrs = [attr[1] for attr in attributions]
    # print(attrs)
    # print(indices_below_threshold)
    collapsing = False
    for i, data in enumerate(zip(attributions, tokens)):
        attr, token = data
        if token in ['[SEP]', '[CLS]', '<s>', '</s>']:
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

    def load_pipelines(self, base_path):
        finetuned_huggingface_model = f"{base_path}{self.huggingface_models[0]}/{self.dataset_name.replace('_bigbio_kb', '')}/final"
        model: AutoModelForTokenClassification = AutoModelForTokenClassification \
            .from_pretrained(finetuned_huggingface_model,
                             local_files_only=True,
                             num_labels=len(self.label2id)).to(CUDA_DEVICE)
        model.config.label2id = self.label2id
        model.config.id2label = self.id2label
        self.pipeline = TokenClassificationPipeline(model=model, tokenizer=self.tokenizers[self.huggingface_models[0]])
        finetuned_other_huggingface_model = f"{base_path}{self.huggingface_models[1]}/{self.dataset_name.replace('_bigbio_kb', '')}/final"
        other_model: AutoModelForTokenClassification = AutoModelForTokenClassification\
            .from_pretrained(finetuned_other_huggingface_model,
                             local_files_only=True,
                             num_labels=len(self.label2id)).to(CUDA_DEVICE)
        other_model.config.label2id = self.label2id
        other_model.config.id2label = self.id2label
        self.other_pipeline = TokenClassificationPipeline(model=other_model, tokenizer=self.tokenizers[self.huggingface_models[1]])

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

    def prepare(self, doc_id, ref1_token_idx, ref2_token_idx):
        self.doc_id = doc_id
        self.ref1_token_idx = ref1_token_idx
        self.ref2_token_idx = ref2_token_idx
        self.entity = None
        for e in self.entities[self.huggingface_models[0]][attribution_types[0]]:
            if e['index'] == ref1_token_idx:
                # print(e['doc_doc_id' if 'doc_doc_id' in e else 'doc_id'] == str(doc_id), 'doc_doc_id' in e, e['doc_doc_id' if 'doc_doc_id' in e else 'doc_id'], e['doc_id'], str(doc_id), e['index'] == ref1_token_idx, e['index'], ref1_token_idx)
                if ('doc_doc_id' in e and e['doc_doc_id'] == str(doc_id)) or e['doc_id'] == str(doc_id):
                    self.entity = e
        assert self.entity is not None
        # self.entity = [e for e in self.entities[self.huggingface_models[0]][attribution_types[0]]
        #                if e['doc_doc_id' if 'doc_doc_id' in e else 'doc_id'] == str(doc_id) and e['index'] == ref1_token_idx][0]
        doc = [doc for doc in self.dataset if doc_id in [doc['document_id'], doc['id']]][0]
        self.docs = {
            'bioelectra-discriminator': self.pre_processors['bioelectra-discriminator'](doc),
            'roberta': self.pre_processors['roberta'](doc)
        }

    def print_table(self, model_i, k_value=5, collapse_threshold=0.05):
        def _get_cell(content, multirow=1):
            return f"\\parbox[b]{{4mm}}{{\\multirow{{{multirow}}}{{*}}{{\\rotatebox[origin=t]{{90}}{{{content}}}}}}} &"

        model = self.huggingface_models[model_i]
        doc = self.docs[model]
        model_string = 'BioElectra' if model.startswith('bioele') else 'RoBERTa'
        first_model_string = 'BioElectra' if self.huggingface_models[0].startswith('bioele') else 'RoBERTa'
        dataset_string = self.dataset_name.replace('_disease', '').replace('_bigbio_kb', '').upper()
        entity_eval = self.entity['eval'] if model_i == 0 else self.other_entity['eval']
        true_class = self.id2label[self.entity['gold_label']]
        if model_i == 0:
            pred_class = self.id2label[self.entity['pred_label']]
        else:
            pred_class = self.id2label[self.other_entity['pred_label']] if 'pred_label' in self.other_entity else '?'
        latex_tables = '''\\begin{table}
\\smaller
\\centering
'''
        latex_tables += f"\\caption{{\\label{{tab:6_example_1}}{model_string} attributions for example x (dataset {dataset_string}, {entity_eval} in {first_model_string}, true class: {true_class}, predicted class: {pred_class})}}\n"
        latex_tables += '''\\toprule
\\begin{tabularx}{\\linewidth}{cc|X@{}}
\\textbf{Attr} & \\textbf{Class} & \\textbf{Text} \\\\'''
        line = 0
        ref_token_idx = self.ref1_token_idx if model_i == 0 else self.ref2_token_idx
        # tokens = self.tokenizers[model].batch_decode(self.docs[model]['input_ids'])
        tokens = self.tokenizers[model].convert_ids_to_tokens(self.docs[model]['input_ids'])
        for attribution_type in self.attribution_types:
            attr_string = attribution_type.upper() if attribution_type != 'gradcam' else 'GradCAM'
            print(model, attribution_type, collapse_threshold)
            entity = [e for e in self.entities[model][attribution_type]
                      if (e.get('doc_doc_id') in [doc.get('document_id'), doc.get('id')] or e.get('doc_id') in [doc.get('document_id'), doc.get('id')]) and e['index'] == ref_token_idx][0]
            for prefix in ['', 'other_']:
                # if prefix == 'other_' and entity['other_entity'] == 'O':
                #     line += 1
                #     continue
                row = '\n'
                if line % 2 == 0:
                    row += '\\midrule\n'
                elif line > 0:
                    row += '\\cmidrule{3-3}\n'
                class_name = CLASS_NAMES[entity[f'{prefix}entity']]
                if not prefix:
                    row += f"{_get_cell(attr_string, 2)} {_get_cell(class_name)}"
                else:
                    row += f"& {_get_cell(class_name)}"
                text = generate_latex_text(
                    entity[f'{prefix}attribution_scores'],
                    tokens,
                    reference_token_idx=entity['index'],
                    rationale_1=entity['rationales'][f'{prefix}top_k'][str(k_value)],
                    collapse_threshold=collapse_threshold,
                    is_roberta=model == 'roberta',
                )
                latex_tables += f"{row} {text} \\\\"
                line += 1
        latex_tables += '''\\bottomrule
\\end{tabularx}
\\end{table}'''
        return latex_tables

    def pick_entities(self, eval_=None, doc_id=None, n_value=1, k_values=[5, 10], allow_zero=False, ref1_token_idx=None):
        # if allow_zero:
        #     allow_zero = random.randint(0, 99) > 66
        filtered_entities = None
        attribution_type = self.attribution_types[0]
        if eval_:
            # test = [i for i in self.entities[self.huggingface_models[0]]]
            # print(test)
            if not ref1_token_idx:
                filtered_entities = list(
                    filter(lambda x: x['eval'] == eval_ and x['entity'].startswith('B'),
                           self.entities[self.huggingface_models[0]][attribution_type]))
                if not allow_zero:
                    filtered_entities = list(
                        filter(lambda x: x['gold_label'] != 0 and x['pred_label'] != 0,
                               filtered_entities))
            else:
                filtered_entities = list(
                    filter(lambda x: (x.get('doc_id') == str(doc_id) or x['doc_doc_id' if 'doc_doc_id' in x else 'doc_id'] == str(doc_id))
                                     and x['index'] == ref1_token_idx,
                           self.entities[self.huggingface_models[0]][attribution_type]))

        else:
            filtered_entities = self.entities[self.huggingface_models[0]][attribution_type]
        if doc_id and not ref1_token_idx:
            self.entity = [e for e in filtered_entities if (e['doc_doc_id' if 'doc_doc_id' in self.entities else 'doc_id'] == str(doc_id)) or e.get('doc_id') == str(doc_id)][0]
        else:
            indices = [i for i in range(len(filtered_entities))]
            # print(len(filtered_entities))
            chosen_entities = []
            for n in range(n_value):
                i = indices.pop(random.choice(indices))
                chosen_entities.append(filtered_entities[i])
            self.entity = chosen_entities[0]

        print(self.entity['eval'], ', pred:', self.id2label[self.entity['pred_label']], ', gold:', self.id2label[self.entity['gold_label']])
        doc_id = self.entity['doc_doc_id' if 'doc_doc_id' in self.entities else 'doc_id']
        doc_id2 = self.entity.get('doc_id')
        idx = self.entity['index']
        # doc_ids = [doc['document_id'] for doc in self.dataset][0:100]
        # print(doc_id, doc_ids)
        # print(self.entity)
        doc = [doc for doc in self.dataset if doc['document_id'] == doc_id]
        if not doc:
            # doc_id = self.entity['doc_id']
            doc = [doc for doc in self.dataset if doc['id'] == doc_id]
        if not doc:
            print(f"doc {self.entity['doc_doc_id' if 'doc_doc_id' in self.entities else 'doc_id']} (id: {self.entity.get('doc_id')}) not found!")
        doc = doc[0]
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
        return idx, tokens[idx], doc_id, doc_id2

    def find_in_other_model(self, model_1_token, reference_token_idx=-1):
        model_1_token = model_1_token.replace('##', '')
        other_model = self.huggingface_models[1]
        other_doc = self.docs[other_model]
        tokens_other_model = self.tokenizers[other_model].batch_decode(other_doc['input_ids'])
        # tokens_other_model = self.tokenizers[other_model].convert_ids_to_tokens(self.docs[other_model]['input_ids'])
        self.other_entity = [e for e in self.entities[other_model][self.attribution_types[0]] if
                             e['doc_doc_id' if 'doc_doc_id' in e else 'doc_id'] == other_doc['document_id'] and e['index'] == reference_token_idx]
        if self.other_entity:
            print(f"Other entity exists: {self.other_entity[0]['eval']}, pred: {self.id2label[self.other_entity[0]['pred_label']]}, gold: {self.id2label[self.other_entity[0]['gold_label']]}")
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

    def ensure_attr_scores_in_models(self, k_value):
        def write_rationale(entity):
            entity['rationales'] = {'top_k': {}, 'other_top_k': {}}
            for prefix in ['', 'other_']:
                # if prefix == 'other_' and entity['other_entity'] in [None, 'O']:
                #     continue
                entity['rationales'][f'{prefix}top_k'][str(k_value)] = get_rationale(entity[f'{prefix}attribution_scores'], k_value)

        model = self.huggingface_models[0]
        other_model = self.huggingface_models[1]
        doc = self.docs[model]
        print(type(doc['document_id']), type(doc['id']))
        other_doc = self.docs[other_model]
        for attr_type in self.attribution_types:
            entity = [e for e in self.entities[model][attr_type] if
                      (e['doc_doc_id' if 'doc_doc_id' in e else 'doc_id'] in [doc['document_id'], doc['id']]
                       or e.get('doc_id') in [doc['document_id'], doc['id']]) and e['index'] == self.ref1_token_idx][0]
            if entity['other_entity'] in [0, 'O', None]:
                print(f'get attributions for class 0 for entity ({attr_type})')
                explainer = TokenClassificationExplainer(self.pipeline.model, self.pipeline.tokenizer, attr_type)
                token_class_index_tuples = [(self.ref1_token_idx, 0)]
                explainer(doc['text'], token_class_index_tuples=token_class_index_tuples,
                          internal_batch_size=BATCH_SIZE)
                word_attributions = explainer.word_attributions
                entity['other_entity'] = self.id2label[0]
                entity['other_attribution_scores'] = word_attributions[self.id2label[0]][self.ref1_token_idx]
                write_rationale(entity)
                idx = self.entities[model][attr_type].index(entity)
                self.entities[model][attr_type][idx] = entity

            self.other_entity = [e for e in self.entities[other_model][attr_type] if
                                 (e['doc_doc_id' if 'doc_doc_id' in e else 'doc_id'] in [other_doc['document_id'], other_doc['id']]
                                  or e.get('doc_id') in [other_doc['document_id'], other_doc['id']]) and e['index'] == self.ref2_token_idx]
            if self.other_entity:
                self.other_entity = self.other_entity[0]
                print('Entity existed already:', self.other_entity['eval'], self.other_entity['other_entity'])
                # print(self.other_entity)
                if self.other_entity['other_entity'] in [0, 'O', None]:
                    main_entity_labels = [self.entity['gold_label'], self.entity['pred_label']]
                    main_other_labels = [self.other_entity['gold_label'], self.other_entity['pred_label']]
                    labels_to_attribute = [label for label in main_entity_labels if label not in main_other_labels or label == 0]
                    # if len(labels_to_attribute) == 0 or len(labels_to_attribute) > 1:
                    #     print('possible error!', main_entity_labels, main_other_labels)
                    # else:
                    print(f'get attributions for other entity ({attr_type})')
                    explainer = TokenClassificationExplainer(self.other_pipeline.model, self.other_pipeline.tokenizer, attr_type)
                    token_class_index_tuples = [(self.ref2_token_idx, labels_to_attribute[0])]
                    explainer(other_doc['text'], token_class_index_tuples=token_class_index_tuples,
                              internal_batch_size=BATCH_SIZE)
                    word_attributions = explainer.word_attributions
                    self.other_entity['other_entity'] = self.id2label[labels_to_attribute[0]]
                    self.other_entity['other_attribution_scores'] = word_attributions[self.id2label[labels_to_attribute[0]]][self.ref2_token_idx]
                    write_rationale(self.other_entity)
                    idx = self.entities[other_model][attr_type].index(self.other_entity)
                    self.entities[other_model][attr_type][idx] = self.other_entity

            else:
                print(f'get attributions {attr_type}')
                token_class_index_tuples = [(self.ref2_token_idx, self.entity['gold_label']), (self.ref2_token_idx, self.entity['pred_label'])]
                explainer = TokenClassificationExplainer(self.pipeline.model, self.pipeline.tokenizer, attr_type)
                explainer(other_doc['text'], token_class_index_tuples=token_class_index_tuples,
                          internal_batch_size=BATCH_SIZE)
                word_attributions = explainer.word_attributions
                self.other_entity = {
                    'eval': 'TN',
                    'index': self.ref2_token_idx,
                    'doc_doc_id': other_doc['document_id'],
                    'entity': self.id2label[self.entity['gold_label']],
                    'attribution_scores': word_attributions[self.id2label[self.entity['gold_label']]][self.ref2_token_idx],
                    'other_entity': self.id2label[self.entity['pred_label']],
                    'other_attribution_scores': word_attributions[self.id2label[self.entity['pred_label']]][self.ref2_token_idx],
                }
                write_rationale(self.other_entity)
                self.entities[other_model][attr_type].append(self.other_entity)

