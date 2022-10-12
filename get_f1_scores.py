import datetime
import json
import math
from argparse import ArgumentParser
from pprint import pprint
import numpy as np
import torch
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
    # print(logits.shape)
    # print(logits)
    predictions = np.argmax(logits, axis=-1)
    # print(predictions)
    labels = map_to_string_vec(labels)
    predictions = map_to_string_vec(predictions)
    metric_scores = metric.compute(predictions=predictions, references=labels)
    return metric_scores


dataset_names = [
    # 'bc5cdr_bigbio_kb',
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

    for i, model_name in enumerate(model_names):
        tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(tokenizers[i])
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

        pre_processor = InputPreProcessor(tokenizer, additional_tokenizers, label2id, max_tokens=512)

        finetuned_huggingface_model = f"./trained_models/{model_name}/{dataset_name.replace('_bigbio_kb', '')}/final"
        model: AutoModelForTokenClassification = AutoModelForTokenClassification \
            .from_pretrained(finetuned_huggingface_model, local_files_only=True, num_labels=len(label2id))
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

        # test = model(torch.tensor([tokenized_datasets[0]['input_ids']]))
        # print(test)
        len_input_ids = [len(doc['input_ids']) for doc in tokenized_datasets]
        # print(len_input_ids)
        input_ids = torch.tensor([doc['input_ids'] for doc in tokenized_datasets])
        # model_predictions = [model(torch.tensor(doc['input_ids'])) for doc in tokenized_datasets]
        model_predictions = model(input_ids)
        gold_references = torch.tensor([doc['labels'] for doc in tokenized_datasets])
        final_score = compute_metrics((model_predictions, gold_references))
        # final_score = metric.compute(predictions=model_predictions, references=gold_references)
        final_score['model_name'] = model_name
        final_score['dataset_name'] = dataset_name

        print(final_score)
