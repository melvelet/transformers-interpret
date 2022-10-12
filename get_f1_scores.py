import datetime
import json
import math
from argparse import ArgumentParser
from pprint import pprint
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline, TrainingArguments, \
    Trainer
from datasets import load_metric

from transformers_interpret.evaluation import InputPreProcessor, NERDatasetAttributor
from transformers_interpret.evaluation.input_pre_processor import get_labels_from_dataset
from bigbio.dataloader import BigBioConfigHelpers


def map_to_string(x):
    return id2label[x]


map_to_string_vec = np.vectorize(map_to_string)
metric = load_metric("seqeval")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    labels = map_to_string_vec(labels)
    predictions = map_to_string_vec(predictions)
    metric_scores = metric.compute(predictions=predictions, references=labels)
    return metric_scores


dataset_names = [
    'bc5cdr_bigbio_kb',
    'ncbi_disease_bigbio_kb',
    'cadec_bigbio_kb',
    'ddi_corpus_bigbio_kb',
]

model_names = [
    'biolinkbert',
    'bioelectra-discriminator',
    'roberta',
]

tokenizers = [
    'michiyasunaga/BioLinkBERT-base',
    'kamalkraj/bioelectra-base-discriminator-pubmed-pmc',
    'Jean-Baptiste/roberta-large-ner-english',
]

for dataset_name in dataset_names:
    conhelps = BigBioConfigHelpers()
    dataset = conhelps.for_config_name(dataset_name).load_dataset()
    label2id, id2label = get_labels_from_dataset(dataset)

    for model_name in model_names:
        tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
        additional_tokenizers = []
        if model_name == 'biolinkbert':
            additional_tokenizers.append(AutoTokenizer.from_pretrained(tokenizers[1]))
            additional_tokenizers.append(AutoTokenizer.from_pretrained(tokenizers[2]))
        elif model_name == 'bioelectra':
            additional_tokenizers.append(AutoTokenizer.from_pretrained(tokenizers[0]))
            additional_tokenizers.append(AutoTokenizer.from_pretrained(tokenizers[2]))
        elif model_name == 'roberta':
            additional_tokenizers.append(AutoTokenizer.from_pretrained(tokenizers[0]))
            additional_tokenizers.append(AutoTokenizer.from_pretrained(tokenizers[1]))

        pre_processor = InputPreProcessor(tokenizer, additional_tokenizers, label2id)

        finetuned_huggingface_model = f"./trained_models/{model_name}/{dataset_name.replace('_bigbio_kb', '')}/final"
        model: AutoModelForTokenClassification = AutoModelForTokenClassification \
            .from_pretrained(finetuned_huggingface_model)
        model.config.label2id = label2id
        model.config.id2label = id2label
        model.config.num_labels = len(id2label)

        if len(dataset) > 1:
            test_dataset = dataset["test"] if 'test' in dataset else dataset["validation"]
        else:
            dataset_length = len(dataset["train"])
            shuffled_dataset = dataset["train"].shuffle(seed=42)
            test_dataset = shuffled_dataset.select(range(math.floor(dataset_length * 0.8), math.floor(dataset_length * 0.9)))
        tokenized_datasets = test_dataset.map(lambda a: pre_processor(a))

        training_args = TrainingArguments(
            output_dir='',
            # label_names=label2id.keys(),
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=1,
            fp16=True,
            save_total_limit=1,
        )
        metric = load_metric("seqeval")

        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            data_collator=transformers.DataCollatorForTokenClassification(tokenizer),
        )

        scores = trainer.evaluate(eval_dataset=test_dataset)

        print(scores)
