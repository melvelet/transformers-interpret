import csv
import math

from bigbio.dataloader import BigBioConfigHelpers
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, \
    TokenClassificationPipeline
from transformers_interpret.evaluation.input_pre_processor import InputPreProcessor, get_labels_from_dataset


def map_to_string(x):
    return id2label[x]


# huggingface_model = 'dbmdz/electra-large-discriminator-finetuned-conll03-english'
# huggingface_model = 'dslim/bert-base-NER'
huggingface_model = 'Jean-Baptiste/roberta-large-ner-english'

tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(huggingface_model)
additional_tokenizers = []

conhelps = BigBioConfigHelpers()
# dataset_name = 'bc5cdr_bigbio_kb'  # 2 classes, short to medium sentence length, Disease
# dataset_name = 'euadr_bigbio_kb'  # 5 classes, short to medium sentence length, Diseases & Disorders
# dataset_name = 'cadec_bigbio_kb'  # 5 classes, shortest documents, forum posts, Disease
# dataset_name = 'scai_disease_bigbio_kb'  # 2 classes, long documents, DISEASE
# dataset_name = 'ncbi_disease_bigbio_kb'
# dataset_name = 'verspoor_2013_bigbio_kb'
dataset_names = [
    'bc5cdr_bigbio_kb',
    # 'euadr_bigbio_kb',
    # 'scai_disease_bigbio_kb',
    # 'ncbi_disease_bigbio_kb',
    # 'verspoor_2013_bigbio_kb',
]

model_name_short = {
    'dbmdz/electra-large-discriminator-finetuned-conll03-english': 'electra',
    'alvaroalon2/biobert_chemical_ner': 'biobert',
    'dslim/bert-base-NER': 'bert',
    'Jean-Baptiste/roberta-large-ner-english': 'roberta',
}

for dataset_name in dataset_names:
    dataset = conhelps.for_config_name(dataset_name).load_dataset()

    label2id, id2label = get_labels_from_dataset(dataset)
    print(label2id)
    pre_processor = InputPreProcessor(tokenizer, additional_tokenizers, label2id)

    tokenized_datasets = dataset.map(lambda a: pre_processor(a))

    finetuned_huggingface_model = ''
    finetuned_huggingface_model = './trained_models/Roberta_BC5'

    model: AutoModelForTokenClassification = AutoModelForTokenClassification.from_pretrained(finetuned_huggingface_model)
    model.config.id2label, model.config.label2id = id2label, label2id
    pipeline = TokenClassificationPipeline(model=model, tokenizer=tokenizer)

    if 'test' in dataset:
        test_dataset = tokenized_datasets["test"]
    else:
        dataset_length = len(dataset["train"])
        shuffled_dataset = tokenized_datasets["train"].shuffle(seed=42)
        test_dataset = shuffled_dataset.select(range(math.floor(dataset_length * 0.8), math.floor(dataset_length * 0.9)))

    for doc_i, doc in enumerate(test_dataset):
        if doc_i < 13:
            continue

        if doc_i >= 20:
            break
        preds = pipeline(doc['text'])
        pred_labels = [0 for _ in range(len(doc['labels']))]
        for i in preds:
            pred_labels[i['index']] = label2id[i['entity']]
        gold_labels = doc['labels']
        #     print(pred_labels, gold_labels)
        for tok_i, labels in enumerate(zip(pred_labels, gold_labels, tokenizer.batch_decode(doc['input_ids']))):
            pred_label = labels[0]
            gold_label = labels[1]
            token_string = labels[2]
            if token_string in ['<pad>', '[PAD]']:
                break
            print(f"{tok_i} {token_string}, {id2label[pred_label]} {'(Gold label: {})'.format(id2label[gold_label]) if pred_label != gold_label else ''}")
        #         if pred_label != gold_label:
        #             print(token_string, id2label[pred_label], 'instead of', id2label[gold_label])
        print(doc['entities'])
        print('\n\n\n')


