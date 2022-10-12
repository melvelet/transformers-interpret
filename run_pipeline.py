import datetime
import json
import math
from argparse import ArgumentParser
from pprint import pprint

import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline

from transformers_interpret.evaluation import InputPreProcessor, NERDatasetEvaluator
from transformers_interpret.evaluation.input_pre_processor import get_labels_from_dataset
from bigbio.dataloader import BigBioConfigHelpers


# k_values = [5]
# k_values =
k_value_levels = [
    [5],
    [2, 3, 5, 10, 20],
    [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20]
    ]
continuous = True
bottom_k = True
evaluate_other = True

attribution_types = [
    'lig',
]

dataset_names = [
    'bc5cdr_bigbio_kb',
    'euadr_bigbio_kb',
    'ncbi_disease_bigbio_kb',
    'scai_disease_bigbio_kb',
    'ddi_corpus_bigbio_kb',
    'mlee_bigbio_kb',
]

huggingface_models = [
    'biolinkbert',
    'bioelectra-discriminator',
    'electra',
    'roberta',
    'bert',
]

parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="model_no", type=int)
parser.add_argument("-d", "--dataset", dest="dataset_no", type=int)
parser.add_argument("-a", "--attribution-type", dest="attribution_type_no", type=int, default=0)
parser.add_argument("-max", "--max-documents", dest="max_documents", type=int, default=0)
parser.add_argument("-s", "--start-document", dest="start_document", type=int, default=0)
parser.add_argument("-k", "--k-value-level", dest="k_value_level", type=int, default=2)
parser.add_argument("-ent", "--entity", dest="entity", type=int, default=0)
args = parser.parse_args()

huggingface_model = huggingface_models[args.model_no]
dataset_name = dataset_names[args.dataset_no]
attribution_type = attribution_types[args.attribution_type_no]
max_documents = args.max_documents
start_document = args.start_document
entity = 'drug' if args.entity == 1 else 'disease'
k_values = k_value_levels[args.k_value_level]
exclude_reference_token = False

print('Loading dataset:', dataset_name)

conhelps = BigBioConfigHelpers()
dataset = conhelps.for_config_name(dataset_name).load_dataset()
doc_ids = [[doc['document_id'] for doc in dataset['train']]]
# print(doc_ids)
# print(dataset['train'][0]['passages'][1]['text'][0])

disease_class = None
if entity == 'disease':
    if dataset_name == 'bc5cdr_bigbio_kb':
        disease_class = 'Disease'
    elif dataset_name == 'euadr_bigbio_kb':
        disease_class = 'Diseases & Disorders'
    elif dataset_name == 'scai_disease_bigbio_kb':
        disease_class = 'DISEASE'
    elif dataset_name == 'ncbi_disease_bigbio_kb':
        disease_class = 'SpecificDisease'
    elif dataset_name == 'verspoor_2013_bigbio_kb':
        disease_class = 'disease'
elif entity == 'drug':
    if dataset_name == 'ddi_corpus_bigbio_kb':
        disease_class = 'DRUG'
    elif dataset_name == 'euadr_bigbio_kb':
        disease_class = 'Chemicals & Drugs'
    elif dataset_name == 'mlee_bigbio_kb':
        disease_class = 'Drug_or_compound'

finetuned_huggingface_model = f"./trained_models/{huggingface_model}/{dataset_name.replace('_bigbio_kb', '')}/final"

model_name_long = {
    'electra': 'dbmdz/electra-large-discriminator-finetuned-conll03-english',
    'bert': 'dslim/bert-base-NER',
    'roberta': 'Jean-Baptiste/roberta-large-ner-english',
    'biolinkbert': 'michiyasunaga/BioLinkBERT-base',
    'bioelectra-discriminator': 'kamalkraj/bioelectra-base-discriminator-pubmed-pmc',
}

print('Loading model:', finetuned_huggingface_model)

tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name_long[huggingface_model])
additional_tokenizers = []
if huggingface_model == 'biolinkbert':
    additional_tokenizers.append(AutoTokenizer.from_pretrained(model_name_long['bioelectra-discriminator']))
elif huggingface_model == 'bioelectra-discriminator':
    additional_tokenizers.append(AutoTokenizer.from_pretrained(model_name_long['biolinkbert']))
model: AutoModelForTokenClassification = AutoModelForTokenClassification.from_pretrained(finetuned_huggingface_model, local_files_only=True)

label2id, id2label = get_labels_from_dataset(dataset)
model.config.label2id = label2id
model.config.id2label = id2label
model.config.num_labels = len(id2label)
pre_processor = InputPreProcessor(tokenizer, additional_tokenizers, label2id, max_tokens=512)
dataset_length = len(dataset["train"])
document_ids = [doc['document_id'] for doc in dataset['train'].shuffle(seed=42)]
# print(document_ids)
if len(dataset) > 1:
    test_dataset = dataset["test"] if 'test' in dataset else dataset["validation"]
else:
    shuffled_dataset = dataset["train"].shuffle(seed=42)
    test_dataset = shuffled_dataset.select(range(math.floor(dataset_length * 0.8), math.floor(dataset_length * 0.9)))
    document_ids = [doc['document_id'] for doc in test_dataset]
    # print(document_ids)
tokenized_datasets = test_dataset.map(lambda a: pre_processor(a))
# print(tokenized_datasets[0]['text'])
document_ids = [doc['document_id'] for doc in tokenized_datasets]
# print('document_ids', document_ids)

base_file_name = f"results/{dataset_name.replace('_bigbio_kb', '')}_{entity}_{huggingface_model}_{attribution_type}"
pipeline = TokenClassificationPipeline(model=model, tokenizer=tokenizer)

with open(f'{base_file_name}_attributed_entities.json', 'r') as f:
    attributions = json.load(f)

evaluator = NERDatasetEvaluator(pipeline, tokenized_datasets, attributions=attributions,
                                attribution_type=attribution_type, class_name=disease_class)
result = evaluator(k_values=k_values,
                   continuous=continuous,
                   bottom_k=bottom_k,
                   max_documents=max_documents,
                   start_document=start_document,
                   evaluate_other=evaluate_other)

pprint(result)

end_time = datetime.datetime.now()


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


exclude_string = 'exclude' if exclude_reference_token else 'include'

with open(f'{base_file_name}_{exclude_string}_{str(end_time).replace(" ", "_")}_scores.json', 'w+') as f:
    json.dump(result, f, cls=NpEncoder)

with open(f'{base_file_name}_{exclude_string}_{str(end_time).replace(" ", "_")}_raw_scores.json', 'w+') as f:
    json.dump(evaluator.raw_scores, f, cls=NpEncoder)

with open(f'{base_file_name}_{exclude_string}_{str(end_time).replace(" ", "_")}_raw_entities.json', 'w+') as f:
    json.dump(evaluator.raw_entities, f, cls=NpEncoder)
