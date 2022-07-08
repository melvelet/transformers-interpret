import os
import numpy as np
from datasets import load_metric
from bigbio.dataloader import BigBioConfigHelpers
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers_interpret.evaluation.input_pre_processor import InputPreProcessor, get_labels_from_dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    print(labels)
    return metric.compute(predictions=predictions, references=labels)


conhelps = BigBioConfigHelpers()
# dataset_name = 'bc5cdr_bigbio_kb'  # 2 classes, short to medium sentence length, Disease
# dataset_name = 'euadr_bigbio_kb'  # 5 classes, short to medium sentence length, Diseases & Disorders
dataset_name = 'cadec_bigbio_kb'  # 5 classes, shortest documents, forum posts, Disease
# dataset_name = 'scai_disease_bigbio_kb'  # 2 classes, long documents, DISEASE
dataset = conhelps.for_config_name(dataset_name).load_dataset()

# huggingface_model = 'Jean-Baptiste/roberta-large-ner-english'
# huggingface_model = 'dbmdz/electra-large-discriminator-finetuned-conll03-english'
# huggingface_model = 'fran-martinez/scibert_scivocab_cased_ner_jnlpba'
# huggingface_model = 'alvaroalon2/biobert_chemical_ner'
huggingface_model = 'dslim/bert-base-NER'
tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(huggingface_model)

label2id, id2label = get_labels_from_dataset(dataset)
print(label2id)
pre_processor = InputPreProcessor(tokenizer, label2id)

model: AutoModelForTokenClassification = AutoModelForTokenClassification.from_pretrained(huggingface_model,
                                                                                         ignore_mismatched_sizes=True,
                                                                                         num_labels=len(label2id))

dataset_length = len(dataset["train"])
tokenized_datasets = dataset.map(lambda a: pre_processor(a))
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(dataset_length//2))
eval_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(dataset_length//2, dataset_length))
print(train_dataset)

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
metric = load_metric("accuracy")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()

os.makedirs(f"trained_models", exist_ok=True)
model.save_pretrained(f"trained_models/{huggingface_model.replace('/', '_')}|{dataset_name}")
