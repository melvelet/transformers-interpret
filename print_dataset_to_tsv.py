import csv
from bigbio.dataloader import BigBioConfigHelpers
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers_interpret.evaluation.input_pre_processor import InputPreProcessor, get_labels_from_dataset


def map_to_string(x):
    return id2label[x]


# huggingface_model = 'dbmdz/electra-large-discriminator-finetuned-conll03-english'
huggingface_model = 'dslim/bert-base-NER'
# huggingface_model = 'Jean-Baptiste/roberta-large-ner-english'

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
    'euadr_bigbio_kb',
    'scai_disease_bigbio_kb',
    'ncbi_disease_bigbio_kb',
    'verspoor_2013_bigbio_kb',
]

for dataset_name in dataset_names:
    dataset = conhelps.for_config_name(dataset_name).load_dataset()

    label2id, id2label = get_labels_from_dataset(dataset)
    print(label2id)
    pre_processor = InputPreProcessor(tokenizer, additional_tokenizers, label2id)

    tokenized_datasets = dataset.map(lambda a: pre_processor(a))

    dataset = tokenized_datasets["train"]

    with open(f'{dataset_name.replace("/", "_")}.tsv', 'w+', newline='') as file:
        tsv_writer = csv.writer(file, delimiter='\t')
        for i, doc in enumerate(dataset):
            tsv_writer.writerow([f'# Document {i}'])
            if i >= 50:
                break
            lines = []
            for token_i, token in enumerate(zip(tokenizer.batch_decode(doc['input_ids']), doc['labels'])):
                if token[0] == '[PAD]':
                    break
                lines.append([token_i, token[0], id2label[token[1]]])
            tsv_writer.writerows(lines)
            tsv_writer.writerow([])
