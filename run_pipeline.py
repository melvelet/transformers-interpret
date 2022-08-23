import datetime
import json
from pprint import pprint
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
from transformers_interpret.evaluation import NERDatasetEvaluator
from bigbio.dataloader import BigBioConfigHelpers

attribution_type = 'lig'
k_values = [5]
# k_values = [2, 3, 5, 10, 20]
# k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
continuous = False
max_documents = None

dataset_name = 'bc5cdr_bigbio_kb'  # 2 classes, short to medium sentence length, Disease
# dataset_name = 'euadr_bigbio_kb'  # 5 classes, short to medium sentence length, Diseases & Disorders
# dataset_name = 'cadec_bigbio_kb'  # 5 classes, shortest documents, forum posts, Disease
# dataset_name = 'scai_disease_bigbio_kb'  # 2 classes, long documents, DISEASE

huggingface_model = 'Jean-Baptiste/roberta-large-ner-english'
# huggingface_model = 'dbmdz/electra-large-discriminator-finetuned-conll03-english'
# huggingface_model = 'fran-martinez/scibert_scivocab_cased_ner_jnlpba'
# huggingface_model = 'alvaroalon2/biobert_chemical_ner'
# huggingface_model = 'dslim/bert-base-NER'
finetuned_huggingface_model = f"trained_models/{huggingface_model.replace('/', '_')}/{dataset_name}/test/"

# model_name_short = {
#     'dbmdz/electra-large-discriminator-finetuned-conll03-english': 'electra',
#     'alvaroalon2/biobert_chemical_ner': 'biobert',
#     'dslim/bert-base-NER': 'bert',
#     'Jean-Baptiste/roberta-large-ner-english': 'roberta',
# }

print('Loading model:', finetuned_huggingface_model)

tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(huggingface_model)
model: AutoModelForTokenClassification = AutoModelForTokenClassification.from_pretrained(finetuned_huggingface_model)

print('Loading dataset:', dataset_name)

conhelps = BigBioConfigHelpers()
dataset = conhelps.for_config_name(dataset_name).load_dataset()

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

pipeline = TokenClassificationPipeline(model=model, tokenizer=tokenizer)
evaluator = NERDatasetEvaluator(pipeline, dataset, attribution_type, class_name=disease_class)

result = evaluator(k_values=k_values, continuous=continuous)

pprint(result)

end_time = datetime.datetime.now()

base_file_name = f"results/{dataset_name}|{huggingface_model.replace('/', '_')}|{attribution_type}|{k_values}|{'cont' if continuous else 'topk'}|{end_time}"

with open(f'{base_file_name}_scores.json', 'w+') as f:
    json.dump(result, f)

with open(f'{base_file_name}_raw_scores.json', 'w+') as f:
    json.dump(str(evaluator.raw_scores), f)

with open(f'{base_file_name}_raw_entities.json', 'w+') as f:
    json.dump(str(evaluator.raw_entities), f)
