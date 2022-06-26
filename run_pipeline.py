import datetime
import json
from pprint import pprint
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline, AutoConfig
from transformers_interpret import TokenClassificationExplainer, MultiLabelClassificationExplainer
from transformers_interpret.evaluation.ner_evaluation import NERSentenceEvaluator, NERDatasetEvaluator
from bigbio.dataloader import BigBioConfigHelpers
from datasets import load_dataset

# huggingface_model = 'Jean-Baptiste/roberta-large-ner-english'
# huggingface_model = 'dbmdz/electra-large-discriminator-finetuned-conll03-english'
# huggingface_model = 'fran-martinez/scibert_scivocab_cased_ner_jnlpba'
# huggingface_model = 'alvaroalon2/biobert_chemical_ner'
huggingface_model = 'dslim/bert-base-NER'

tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(huggingface_model)
model: AutoModelForTokenClassification = AutoModelForTokenClassification.from_pretrained(huggingface_model)

dataset_name = 'bc5cdr_bigbio_kb'
conhelps = BigBioConfigHelpers()
load_dataset_kwargs = conhelps.for_config_name(dataset_name).get_load_dataset_kwargs(data_dir=dataset_name)
dataset = [load_dataset(**load_dataset_kwargs, split='train'), load_dataset(**load_dataset_kwargs, split='test'),
           load_dataset(**load_dataset_kwargs, split='validation')]

pipeline = TokenClassificationPipeline(model=model, tokenizer=tokenizer)
evaluator = NERDatasetEvaluator(pipeline, dataset)

result = evaluator(k_values=[2])

print(result)

with open(f'results/{dataset_name}_{str(datetime.datetime.now())}.json', 'w+') as f:
    json.dumps(result)

with open(f'results/{dataset_name}_{str(datetime.datetime.now())}_scores.json', 'w+') as f:
    json.dumps(evaluator.scores)
