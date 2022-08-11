from transformers import AutoTokenizer

from evaluation.input_pre_processor import InputPreProcessor

BERT_TOKENIZER = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
INPUT_SEQ = 'Recognizing biomedical entities (NER) such as genes, chemicals or diseases in unstructured scientific ' \
            'text is a crucial step of all biomedical information extraction pipelines. The respective tools are ' \
            'typically trained and evaluated on rather small gold standard datasets. However, in any real application ' \
            'they are applied ‘in the wild’, i.e. to a large collection of texts often varying in focus, ' \
            'entity distribution, genre (e.g. patents versus scientific articles) and text type (e.g. abstract versus ' \
            'full text). This mismatch can lead to severely misleading evaluation results. To address this, ' \
            'we recently released the HUNER tagger (Weber et al., 2020) that was trained jointly on a large ' \
            'collection of biomedical NER datasets, leading to a much better performance on unseen corpora compared ' \
            'to models trained on a single corpus. However, HUNER relies on a Docker installation and uses a ' \
            'client-server architecture. These design decisions do not hinder its own installation but make its ' \
            'integration into any of the major NLP frameworks, which is required for the construction of ' \
            'comprehensive information extraction pipelines, cumbersome. Moreover, HUNER does not build upon a ' \
            'pretrained language model (LM), although such models were the basis for many recent breakthroughs in NLP ' \
            'research (Akbik et al., 2019). Here, we present HunFlair, a redesigned and retrained version of HUNER ' \
            'integrated into the widely used Flair NLP framework. HunFlair builds upon a pretrained character-level ' \
            'language model. It recognizes five important biomedical entity types with high accuracy, namely Cell ' \
            'Lines, Chemicals, Diseases, Genes and Species. Through its shipping as a Flair component, ' \
            'it can be easily combined with other IE tools (e.g. text parsing, document classification, ' \
            'hedge detection) or other language models and benefits from the experiences and future developments of ' \
            'the large user and developer base of Flair. Through its simple but extensible interface, it is easily ' \
            'accessible also for non-experts. Technically, HunFlair combines the insights from Weber et al. (2020) ' \
            'and Akbik et al. (2019) by merging character-level LM pretraining and joint training on multiple gold ' \
            'standard corpora, which leads to strong gains over other state-of-the-art off-the-shelf NER tools. For ' \
            'HunFlair, we specially trained a character-level in-domain LM on a large corpus of biomedical abstracts ' \
            'and full-texts and make it publicly available to facilitate further research. In addition, we integrate ' \
            '23 biomedical NER corpora into HunFlair using a consistent format, which enables researchers and ' \
            'practitioners to rapidly train their own models and experiment with new approaches within Flair. Note ' \
            'that these are the same corpora that were already made available through HUNER. However, the integration ' \
            'into Flair has the additional benefits of more convenient automated downloading and flexible ' \
            'preprocessing. While HUNER’s corpora came preprocessed with a particular method, users of HunFlair may ' \
            'process the corpora along with their own choices, for instance by using different sentence resp. word ' \
            'segmentation methods. HunFlair was created by implementing the approach behind HUNER into the Flair NLP ' \
            'framework, along with its improvement by integrating a pretrained language model. Flair is an NLP ' \
            'framework designed to allow intuitive training and distribution of sequence labeling, ' \
            'text classification and language models. Flair achieves state-of-the-art performance in several NLP ' \
            'research challenges (Akbik et al., 2018), allows researchers to ‘mix and match’ various types of ' \
            'character, word and document embeddings and features a base of more than 120 contributors. In addition, ' \
            'more than 500 open-source projects and python libraries rely on Flair.'


def test_truncate_to_512_tokens():
    input_length = len(BERT_TOKENIZER(INPUT_SEQ)['input_ids'])
    sut = InputPreProcessor(BERT_TOKENIZER, None, max_tokens=512)
    result = sut(INPUT_SEQ)
    result_length = len(BERT_TOKENIZER(result)['input_ids'])
    assert input_length != result_length
    assert result_length <= 512
