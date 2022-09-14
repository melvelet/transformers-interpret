import csv

from bigbio.dataloader import BigBioConfigHelpers
from transformers import AutoTokenizer

from transformers_interpret.evaluation import InputPreProcessor
from transformers_interpret.evaluation.input_pre_processor import get_labels_from_dataset

conhelps = BigBioConfigHelpers()
dataset_names = [
    'bc5cdr_bigbio_kb',
    'euadr_bigbio_kb',
    'scai_disease_bigbio_kb',
    'ncbi_disease_bigbio_kb',
    'verspoor_2013_bigbio_kb',
]

huggingface_models = ('dslim/bert-base-NER', 'dbmdz/electra-large-discriminator-finetuned-conll03-english', 'Jean-Baptiste/roberta-large-ner-english')

lines = [['dataset', 'model', 'total_documents', 'truncated_documents', 'truncated_documents_percentage',
          'total_entities', 'remaining_entities', 'truncated_entities', 'truncated_entities_percentage',
          'total_tokens', 'truncated_tokens', 'truncated_tokens_percentage']]

for model in huggingface_models:
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model)
    additional_tokenizers = []
    for dataset_name in dataset_names:
        dataset = conhelps.for_config_name(dataset_name).load_dataset()
        label2id, id2label = get_labels_from_dataset(dataset)
        pre_processor = InputPreProcessor(tokenizer, additional_tokenizers, label2id)
        truncated_tokens, truncated_documents, truncated_entities, remaining_entities, total_tokens, total_documents = 0, 0, 0, 0, 0, 0
        for split in dataset:
            total_documents += len(dataset[split])
            for document in dataset[split]:
                pre_processed_document = pre_processor(document)
                truncated_tokens += pre_processor.stats['truncated_tokens']
                total_tokens += pre_processor.stats['total_tokens']
                truncated_documents += 1 if pre_processor.stats['is_truncated'] > 0 else 0
                truncated_entities += pre_processor.stats['truncated_entities']
                remaining_entities += pre_processor.stats['remaining_entities']
        lines.append([dataset_name, model, total_documents, truncated_documents, truncated_documents / total_documents,
                      truncated_entities + remaining_entities, remaining_entities, truncated_entities,
                      truncated_entities / (truncated_entities + remaining_entities),
                      total_tokens, truncated_tokens, truncated_tokens / total_tokens])

with open('truncation_stats.csv', 'w+') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(lines)
