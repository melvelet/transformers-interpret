from pprint import pprint

import spacy
import torch


def get_labels_from_dataset(dataset):
    seen_labels = list(set(
        [entity['type'] for split in dataset for document in dataset[split] for entity in document['entities']]
    ))
    labels = ['O']
    labels.extend(
        sorted(seen_labels)
    )

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {y: x for x, y in label2id.items()}

    return label2id, id2label


class InputPreProcessor:
    def __init__(self, tokenizer, label2id, max_tokens: int = 512):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.sentence_segmentation = spacy.load("en_core_web_sm")
        self.label2id = label2id

    def __call__(self, input_document):
        raw_input_text = ''. join([i[0] for i in [passage['text'] for passage in input_document['passages']]])\
            .replace('(ABSTRACT TRUNCATED AT 250 WORDS)', '')
        result_text, truncated_tokens = self.truncate_input(raw_input_text)
        tokens = self.tokenizer(result_text, padding='max_length', return_offsets_mapping=True)
        labels = self.create_labels(input_document, tokens)
        result_document = {
            'id': input_document['id'],
            'document_id': input_document['document_id'],
            'text': result_text,
            'labels': labels,
        }
        result_document.update(tokens)
        self.stats = {
            'truncated_tokens': truncated_tokens,
            'is_truncated': truncated_tokens > 0,
            'annotated_entities': len(input_document['entities']),
        }
        return result_document

    def create_labels(self, document, tokens):
        def _check_for_offset_overlap(tok_offset, ent_offset):
            return tok_offset[0] != tok_offset[1]\
                   and tok_offset[0] >= ent_offset[0] and tok_offset[1] <= ent_offset[1]

        labels = []
        for token_idx, token_offset in enumerate(tokens['offset_mapping']):
            label = self.label2id['O']
            for entity in document['entities']:
                entity_class = entity['type']
                entity_offset = entity['offsets'][0]
                if _check_for_offset_overlap(token_offset, entity_offset):
                    label = self.label2id[entity_class]
            labels.append(label)

        return torch.IntTensor(labels)

    def truncate_input(self, raw_input_text):
        if not raw_input_text:
            return '', 0
        sentences = list(self.sentence_segmentation(raw_input_text).sents)
        tokens = []
        included_sentences = []
        truncated_tokens = 0
        for i, sent in enumerate(sentences):
            # print(i, str(sent))
            input_tokens_sent = self.tokenizer.tokenize(str(sent))
            if len(tokens) + len(input_tokens_sent) <= self.max_tokens - 2 and truncated_tokens == 0:
                tokens.extend(input_tokens_sent)
                included_sentences.append(str(sent))
            else:
                truncated_tokens += len(input_tokens_sent)
        try:
            assert len(included_sentences) > 0
        except AssertionError as e:
            print('AssertionError!', raw_input_text)
        result_document = ' '.join(included_sentences)
        return result_document, truncated_tokens
