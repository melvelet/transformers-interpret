import csv
import math
import os
import traceback
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
    return metric.compute(predictions=predictions, references=labels)


conhelps = BigBioConfigHelpers()
# dataset_name = 'bc5cdr_bigbio_kb'  # 2 classes, short to medium sentence length, Disease
# dataset_name = 'euadr_bigbio_kb'  # 5 classes, short to medium sentence length, Diseases & Disorders
# dataset_name = 'cadec_bigbio_kb'  # 5 classes, shortest documents, forum posts, Disease
# dataset_name = 'scai_disease_bigbio_kb'  # 2 classes, long documents, DISEASE
dataset_names = [
    'bc5cdr_bigbio_kb',
    # 'euadr_bigbio_kb',
    'cadec_bigbio_kb',
    'scai_disease_bigbio_kb',
]

# huggingface_model = 'Jean-Baptiste/roberta-large-ner-english'
# huggingface_model = 'dbmdz/electra-large-discriminator-finetuned-conll03-english'
# huggingface_model = 'fran-martinez/scibert_scivocab_cased_ner_jnlpba'
# huggingface_model = 'alvaroalon2/biobert_chemical_ner'
# huggingface_model = 'dslim/bert-base-NER'
huggingface_models = [
    'dbmdz/electra-large-discriminator-finetuned-conll03-english',
    'fran-martinez/scibert_scivocab_cased_ner_jnlpba',
    'alvaroalon2/biobert_chemical_ner',
    'dslim/bert-base-NER',
    'Jean-Baptiste/roberta-large-ner-english',
]

csv_data = [['dataset_name'] + huggingface_models]
for dataset_name in dataset_names:
    dataset_scores = [dataset_name]
    dataset = conhelps.for_config_name(dataset_name).load_dataset()

    for huggingface_model in huggingface_models:
        try:
            tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(huggingface_model)

            label2id, id2label = get_labels_from_dataset(dataset)
            print(label2id)
            pre_processor = InputPreProcessor(tokenizer, label2id)

            model: AutoModelForTokenClassification = AutoModelForTokenClassification\
                .from_pretrained(huggingface_model,
                                 ignore_mismatched_sizes=True,
                                 num_labels=len(label2id))
            tokenized_datasets = dataset.map(lambda a: pre_processor(a))
            map_to_string_vec = np.vectorize(map_to_string)

            dataset_length = len(dataset["train"])
            if len(dataset) > 1:
                train_dataset = tokenized_datasets["train"]
                eval_dataset = tokenized_datasets["test"]
            else:
                train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(math.floor(dataset_length * 0.8)))
                eval_dataset = tokenized_datasets["train"].shuffle(seed=42) \
                    .select(range(math.floor(dataset_length * 0.8), dataset_length))
            print('train_dataset', train_dataset)
            print('eval_dataset', eval_dataset)

            training_args = TrainingArguments(
                output_dir="trained_models",
                # label_names=label2id.keys(),
                evaluation_strategy="epoch",
                save_strategy="epoch",
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                num_train_epochs=10,
                learning_rate=5e-05,
                warmup_ratio=0.04,
                metric_for_best_model="overall_f1",
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
            scores = trainer.evaluate()
            dataset_scores.append(scores['eval_overall_f1'])

            os.makedirs(f"trained_models/{huggingface_model.replace('/', '_')}", exist_ok=True)
            # model.save_pretrained(f"trained_models/{huggingface_model.replace('/', '_')}/{dataset_name}.pth")

        except Exception as e:
            dataset_scores.append(traceback.format_exc())

    csv_data.append(dataset_scores)

    with open('dataset_model_scores.csv', 'w+', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(csv_data)
