import spacy
import torch


class InputPreProcessor:
    def __init__(self, tokenizer, max_tokens: int = 512):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.sentence_segmentation = spacy.load("en_core_web_sm")

    def __call__(self, input_document):
        raw_input_text = ''. join([i[0] for i in [passage['text'] for passage in input_document['passages']]])\
            .replace('(ABSTRACT TRUNCATED AT 250 WORDS)', '')
        result_document, truncated_tokens = self.truncate_input(raw_input_text)
        tokens = self.tokenizer(result_document, return_offsets_mapping=True)
        labels = self.create_labels(input_document, tokens)
        self.stats = {
            'truncated_tokens': truncated_tokens,
            'is_truncated': truncated_tokens > 0,
        }
        return result_document

    @staticmethod
    def create_labels(document, tokens):
        def _check_for_offset_overlap(token_offset, entity_offset):
            return token_offset[0] != token_offset[1]\
                   and token_offset[0] >= entity_offset[0] and token_offset[1] <= entity_offset[1]

        labels = []
        texts = []
        for token_idx, tok_offset in enumerate(tokens['offset_mapping']):
            label = 'O'
            text = ''
            for entity in document['entities']:
                entity_class = entity['type']
                entity_text = entity['text'][0]
                entity_offset = entity['offsets'][0]
                if _check_for_offset_overlap(tok_offset, entity_offset):
                    label = entity_class
                    text = entity_text
            labels.append(label)
            texts.append(text)

        return torch.IntTensor(labels)

    def truncate_input(self, raw_input_text):
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
        assert len(included_sentences) > 0
        result_document = ' '.join(included_sentences)
        return result_document, truncated_tokens

    @staticmethod
    def get_labels_from_dataset(dataset):
        seen_labels = list(set([entity['type'] for split in dataset for document in dataset[split] for entity in document['entities']]))
        labels = ['O']
        labels.extend(
            sorted(seen_labels)
        )

        label2id = {label: i for i, label in enumerate(labels)}
        print(label2id)

        return label2id
