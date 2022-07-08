import os
import numpy as np
from datasets import load_metric
from bigbio.dataloader import BigBioConfigHelpers
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers_interpret.evaluation.input_pre_processor import InputPreProcessor, get_labels_from_dataset


def compute_metrics(eval_pred):
    print(eval_pred, eval_pred.shape)
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


conhelps = BigBioConfigHelpers()
dataset_name = 'cadec_bigbio_kb'
dataset = conhelps.for_config_name(dataset_name).load_dataset()

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
