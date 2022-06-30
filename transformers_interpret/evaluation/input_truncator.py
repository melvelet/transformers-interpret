import spacy
from spacy.pipeline.senter import DEFAULT_SENTER_MODEL


class InputTruncator:
    def __init__(self, tokenizer, max_tokens: int = 512):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.sentence_segmentation = spacy.load("en_core_web_sm")

    def __call__(self, input_seq: str):
        sentences = list(self.sentence_segmentation(input_seq).sents)
        tokens = []
        included_sentences = []
        for i, sent in enumerate(sentences):
            input_tokens_sent = self.tokenizer.tokenize(str(sent))
            if len(tokens) + len(input_tokens_sent) <= self.max_tokens - 2:
                tokens.extend(input_tokens_sent)
                included_sentences.append(str(sent))
            else:
                break
        assert len(included_sentences) > 0
        result_document = ' '.join(included_sentences)
        print(result_document)
        return result_document
