import spacy
import torch


def get_labels_from_dataset(dataset, has_splits=True):
    if has_splits:
        seen_labels = list(set(
            [entity['type'] for split in dataset for document in dataset[split] for entity in document['entities']]
        ))
    else:
        seen_labels = list(set(
            [entity['type'] for document in dataset for entity in document['entities']]
        ))
    seen_labels.sort()
    labels = ['O']
    for label in seen_labels:
        if label != '':
            labels.append(f"B-{label}")
            labels.append(f"I-{label}")

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {y: x for x, y in label2id.items()}

    return label2id, id2label


class InputPreProcessor:
    def __init__(self, tokenizer, additional_tokenizers, label2id, max_tokens: int = 512):
        self.tokenizer = tokenizer
        self.additional_tokenizers = additional_tokenizers if additional_tokenizers else []
        self.max_tokens = max_tokens
        self.sentence_segmentation = spacy.load("en_core_web_sm")
        self.label2id = label2id
        self.id2label = {y: x for x, y in self.label2id.items()}

    def __call__(self, input_document):
        raw_input_text = ' '.join([i[0] for i in [passage['text'] for passage in input_document['passages']]])\
            .replace('(ABSTRACT TRUNCATED AT 250 WORDS)', '')
        result_text, truncated_tokens, cutoff_index = self.truncate_input(raw_input_text)
        if len(result_text) == 0:
            print(input_document)
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
        truncated_entities = len([i for i in input_document['entities'] if i['offsets'][0][0] >= cutoff_index])
        self.stats = {
            'truncated_tokens': truncated_tokens,
            'total_tokens': len(result_document['input_ids']),
            'is_truncated': truncated_tokens > 0,
            'annotated_entities': len(input_document['entities']),
            'remaining_entities': len(input_document['entities']) - truncated_entities,
            'truncated_entities': truncated_entities,
        }
        return result_document

    def create_labels(self, document, tokens):
        def _check_for_offset_overlap(tok_offset, ent_offset):
            return tok_offset[0] != tok_offset[1]\
                   and tok_offset[0] >= ent_offset[0] and tok_offset[1] <= ent_offset[1]

        labels = []
        label_previous = ''
        highest_offset = max([i[1] for i in tokens['offset_mapping']])
        previous_entity_id = -1
        for token_idx, token in enumerate(zip(tokens['offset_mapping'], tokens['input_ids'])):
            token_offset, token_id = token
            label = 'O'
            token_text = self.tokenizer.decode(token_id).replace('##', '').replace('Ġ', '')
            current_entity_id = -1
            for entity in document['entities']:
                entity_class = entity['type']
                entity_offset = entity['offsets'][0]
                entity_text = entity['text'][0]
                entity_id = entity['id']
                if entity['type'] and _check_for_offset_overlap(token_offset, entity_offset):
                    label = entity_class
                    current_entity_id = entity_id
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

            if label == 'O' or label == '':
                labels.append(self.label2id['O'])
            elif label_previous == label and current_entity_id == previous_entity_id:
                labels.append(self.label2id[f"I-{label}"])
            else:
                labels.append(self.label2id[f"B-{label}"])
            label_previous = label
            previous_entity_id = current_entity_id

        return torch.IntTensor(labels)

    def truncate_input(self, raw_input_text):
        if not raw_input_text:
            return '', 0, 0
        split_sentences = self.sentence_segmentation(raw_input_text)
        sentences = list(split_sentences.sents)
        no_additional_tokenizers = len([self.tokenizer] + self.additional_tokenizers)
        tokens_temp = [[] for _ in range(no_additional_tokenizers)]
        cutoff_index_temp = [len(raw_input_text) for _ in range(no_additional_tokenizers)]
        truncated_tokens_temp = [0] * no_additional_tokenizers
        for tokenizer_i, tokenizer in enumerate([self.tokenizer] + self.additional_tokenizers):
            for i, sent in enumerate(sentences):
                input_tokens_sent = tokenizer.tokenize(str(sent))
                if len(tokens_temp[tokenizer_i]) + len(input_tokens_sent) < self.max_tokens - 2 and truncated_tokens_temp[tokenizer_i] == 0:
                    tokens_temp[tokenizer_i].extend(input_tokens_sent)
                else:
                    if truncated_tokens_temp[tokenizer_i] == 0:
                        cutoff_index_temp[tokenizer_i] = sent.start_char
                    truncated_tokens_temp[tokenizer_i] += len(input_tokens_sent)

        tokenizer_i_with_most_cutoff = cutoff_index_temp.index(min(cutoff_index_temp))
        cutoff_index = cutoff_index_temp[tokenizer_i_with_most_cutoff]
        truncated_tokens = truncated_tokens_temp[tokenizer_i_with_most_cutoff]
        result_document = raw_input_text[:cutoff_index]
        return result_document, truncated_tokens, cutoff_index
