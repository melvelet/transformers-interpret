import spacy
import torch


def get_labels_from_dataset(dataset):
    seen_labels = list(set(
        [entity['type'] for split in dataset for document in dataset[split] for entity in document['entities']]
    ))
    seen_labels.sort()
    labels = ['O']
    for label in seen_labels:
        labels.append(f"B-{label}")
        labels.append(f"I-{label}")

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {y: x for x, y in label2id.items()}

    return label2id, id2label


class InputPreProcessor:
    def __init__(self, tokenizer, label2id, max_tokens: int = 512):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.sentence_segmentation = spacy.load("en_core_web_sm")
        self.label2id = label2id
        self.id2label = {y: x for x, y in self.label2id.items()}

    def __call__(self, input_document):
        raw_input_text = ' '.join([i[0] for i in [passage['text'] for passage in input_document['passages']]])\
            .replace('(ABSTRACT TRUNCATED AT 250 WORDS)', '')
        result_text, truncated_tokens = self.truncate_input(raw_input_text)
        tokens = self.tokenizer(result_text, padding='max_length', return_offsets_mapping=True)
        labels = self.create_labels(input_document, tokens)
        # tokens_str = self.tokenizer.tokenize(result_text)
        # for t, l in zip(tokens_str, labels[1:]):
        #     print(t, self.id2label[l.item()])
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
        label_previous = ''
        highest_offset = max([i[1] for i in tokens['offset_mapping']])
        for token_idx, token in enumerate(zip(tokens['offset_mapping'], tokens['input_ids'])):
            token_offset, token_id = token
            label = 'O'
            token_text = self.tokenizer.decode(token_id).replace('##', '').replace('Ġ', '')
            for entity in document['entities']:
                entity_class = entity['type']
                entity_offset = entity['offsets'][0]
                entity_text = entity['text'][0]
                if entity['type'] and _check_for_offset_overlap(token_offset, entity_offset):
                    label = entity_class
                    # if token_text not in entity_text:
                    #     raw_input_text = ' '.join([i[0] for i in [passage['text'] for passage in document['passages']]])
                    #     input_text, _ = self.truncate_input(raw_input_text)
                    #     print(document['id'], 'token_text:', token_text, 'entity_text:', entity_text)
                    #     print('token_offset:', token_offset, 'entity_offset:', entity_offset, 'highest:', highest_offset,
                    #           'offset:', document['passages'][1]['offsets'][0][1], 'len(input_test):', len(input_text))
                    #     # print(differ.compare(input_text, raw_input_text))
                    #     # print(document['passages'])
                    #     # raise Exception('Incorrect label matching')
                    break

            if label == 'O':
                labels.append(self.label2id['O'])
            elif label_previous == label:
                labels.append(self.label2id[f"I-{label}"])
            else:
                labels.append(self.label2id[f"B-{label}"])
            label_previous = label

        return torch.IntTensor(labels)

    def truncate_input(self, raw_input_text):
        if not raw_input_text:
            return '', 0
        split_sentences = self.sentence_segmentation(raw_input_text)
        sentences = list(split_sentences.sents)
        tokens = []
        cutoff_index = len(raw_input_text)
        truncated_tokens = 0
        for i, sent in enumerate(sentences):
            input_tokens_sent = self.tokenizer.tokenize(str(sent))
            if len(tokens) + len(input_tokens_sent) - 2 <= self.max_tokens - 2 and truncated_tokens == 0:
                tokens.extend(input_tokens_sent)
            else:
                cutoff_index = sent.start_char
                truncated_tokens += len(input_tokens_sent)
        result_document = raw_input_text[:cutoff_index]
        return result_document, truncated_tokens
