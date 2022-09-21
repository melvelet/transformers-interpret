import math
import os
from argparse import ArgumentParser

import numpy as np
from datasets import load_metric
from bigbio.dataloader import BigBioConfigHelpers
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers_interpret.evaluation.input_pre_processor import InputPreProcessor, get_labels_from_dataset


def map_to_string(x):
    return id2label[x]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    labels = map_to_string_vec(labels)
    predictions = map_to_string_vec(predictions)
    metric_scores = metric.compute(predictions=predictions, references=labels)
    metric_scores['target_f1'] = metric_scores[disease_class]['f1']
    return metric_scores


model_name_short = {
    'dbmdz/electra-large-discriminator-finetuned-conll03-english': 'electra',
    'alvaroalon2/biobert_chemical_ner': 'biobert',
    'dslim/bert-base-NER': 'bert',
    'Jean-Baptiste/roberta-large-ner-english': 'roberta',
    'kamalkraj/BioELECTRA-PICO': 'bioelectra',
}

# batch_size = 4
# learning_rate = 5e-05

conhelps = BigBioConfigHelpers()
# dataset_name = 'bc5cdr_bigbio_kb'  # 2 classes, short to medium sentence length, Disease
# dataset_name = 'euadr_bigbio_kb'  # 5 classes, short to medium sentence length, Diseases & Disorders
# dataset_name = 'cadec_bigbio_kb'  # 5 classes, shortest documents, forum posts, Disease
# dataset_name = 'scai_disease_bigbio_kb'  # 2 classes, long documents, DISEASE
# dataset_name = 'ncbi_disease_bigbio_kb'
# dataset_name = 'verspoor_2013_bigbio_kb'
dataset_names = [
    'bc5cdr_bigbio_kb',
    'euadr_bigbio_kb',
    'ncbi_disease_bigbio_kb',
    'scai_disease_bigbio_kb',
]

# huggingface_model = 'dbmdz/electra-large-discriminator-finetuned-conll03-english'
# huggingface_model = 'fran-martinez/scibert_scivocab_cased_ner_jnlpba'
# huggingface_model = 'alvaroalon2/biobert_chemical_ner'
# huggingface_model = 'dslim/bert-base-NER'
# huggingface_model = 'Jean-Baptiste/roberta-large-ner-english'
# huggingface_model = 'kamalkraj/BioELECTRA-PICO'
huggingface_models = [
    'dbmdz/electra-large-discriminator-finetuned-conll03-english',
    'Jean-Baptiste/roberta-large-ner-english',
    'fran-martinez/scibert_scivocab_cased_ner_jnlpba',
    'alvaroalon2/biobert_chemical_ner',
    'dslim/bert-base-NER',
]

parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="model_no", type=int)
parser.add_argument("-d", "--dataset", dest="dataset_no", type=int)
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, default=4)
parser.add_argument("-l", "--learning-rate", dest="learning_rate", type=int, default=1)
args = parser.parse_args()

huggingface_model = huggingface_models[args.model_no]
dataset_name = dataset_names[args.dataset_no]
batch_size = args.batch_size
if args.learning_rate == 0:
    learning_rate = 1e-05
elif args.learning_rate == 2:
    learning_rate = 1e-06
else:
    learning_rate = 5e-05

print('huggingface_model', huggingface_model)
print('dataset_name', dataset_name)
print('batch_size', batch_size)
print('learning_rate', learning_rate)

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
else:
    disease_class = None

# csv_data = [['dataset_name'] + huggingface_models]
# dataset_scores = [dataset_name]
dataset = conhelps.for_config_name(dataset_name).load_dataset()

tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(huggingface_model)
additional_tokenizers = []
if model_name_short[huggingface_model] == 'roberta':
    additional_tokenizers.append(AutoTokenizer.from_pretrained('dbmdz/electra-large-discriminator-finetuned-conll03-english'))
elif model_name_short[huggingface_model] in ['electra', 'bioelectra']:
    additional_tokenizers.append(AutoTokenizer.from_pretrained('Jean-Baptiste/roberta-large-ner-english'))

label2id, id2label = get_labels_from_dataset(dataset)
print(label2id)
pre_processor = InputPreProcessor(tokenizer, additional_tokenizers, label2id)

model: AutoModelForTokenClassification = AutoModelForTokenClassification\
    .from_pretrained(huggingface_model,
                     ignore_mismatched_sizes=True,
                     num_labels=len(label2id))
model.config.label2id = label2id
model.config.id2label = id2label
tokenized_datasets = dataset.map(lambda a: pre_processor(a))
map_to_string_vec = np.vectorize(map_to_string)

dataset_length = len(dataset["train"])
if len(dataset) > 1:
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]
    test_dataset = tokenized_datasets["test"] if 'test' in tokenized_datasets else tokenized_datasets["validation"]
else:
    shuffled_dataset = tokenized_datasets["train"].shuffle(seed=42)
    train_dataset = shuffled_dataset.select(range(math.floor(dataset_length * 0.8)))
    test_dataset = shuffled_dataset.select(range(math.floor(dataset_length * 0.8), math.floor(dataset_length * 0.9)))
    eval_dataset = shuffled_dataset.select(range(math.floor(dataset_length * 0.9), dataset_length))
print('train_dataset', train_dataset)
print('eval_dataset', eval_dataset)
print('test_dataset', test_dataset)


output_dir = f"trained_models/{model_name_short[huggingface_model]}/{dataset_name.replace('_bigbio_kb', '')}"
os.makedirs(output_dir, exist_ok=True)
training_args = TrainingArguments(
    output_dir=output_dir,
    # label_names=label2id.keys(),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    learning_rate=learning_rate,
    warmup_ratio=0.04,
    metric_for_best_model="target_f1",
    load_best_model_at_end=True,
    greater_is_better=True,
)
metric = load_metric("seqeval")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
scores = trainer.evaluate(eval_dataset=test_dataset)
score = scores['eval_overall_f1']
print('dataset_name', dataset_name, 'huggingface_model', huggingface_model, 'f1', scores['eval_overall_f1'])

disease_score = scores[f'eval_{disease_class}']['f1'] if disease_class else 0

trainer.save_model(f"{output_dir}/score{score}_disease{disease_score}_batch{batch_size}_learn{learning_rate}")

# csv_data.append(dataset_scores)
#
# with open('dataset_model_scores.csv', 'w+', encoding='utf-8', newline='') as f:
#     csv_writer = csv.writer(f)
#     csv_writer.writerows(csv_data)
