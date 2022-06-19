import csv

from bigbio.dataloader import BigBioConfigHelpers
from bigbio.utils.constants import Tasks, Lang


conhelps = BigBioConfigHelpers()
ner = Tasks.NAMED_ENTITY_RECOGNITION
eng = Lang.EN

ner_datasets = conhelps.filtered(
    lambda x:
    x.is_bigbio_schema
    and ner in x.tasks
    and eng in x.languages
    and not x.is_broken
)

print("found {} dataset configs from {} datasets".format(
    len(ner_datasets),
    len(ner_datasets.available_dataset_names)
))

with open('/datasets.csv', 'w+', newline='', encoding='utf-8') as file:
    csv_writer = csv.writer(file)
    header = [
        'dataset_name',
        'config_name',
        'description',
        'number_entities',
        'entities',
        'entities_count',
        'passages_count',
        'passages_char_count',
        'mean_passages_char_count',
        'number_splits',
        'splits',
        ]
    csv_writer.writerow(header)
    for data in ner_datasets:
        dataset_name = data.dataset_name
        config_name = data.config.name
        description = data.description
        entities_type_counter = []
        passages_count = 0
        passages_char_count = 0
        entities_count = 0
        splits = []
        try:
            metadata = data.get_metadata()
            splits = [split for split in metadata]
            for _, split in metadata.items():
                entities_type_counter.extend(split.entities_type_counter.keys())
                passages_count += split.passages_count
                passages_char_count += split.passages_char_count
                entities_count += split.entities_count
        except Exception as exc:
            try:
                dataset = data.load_dataset()
                splits = [i for i in dataset]
                for split in splits:
                    dataset = data.load_dataset(split=split)
                    passages_count += len(dataset)
                    for doc in dataset:
                        passages_char_count += len(doc['text'])
                        entities_count += len(doc['entities'])
                        for e in doc['entities']:
                            if e['type'] not in entities_type_counter:
                                entities_type_counter.append(e['type'])
            except Exception as exc_2:
                print('error!', config_name)

        entities_type_counter = set(entities_type_counter)
        line = [
            dataset_name,
            config_name,
            description,
            len(entities_type_counter),
            ', '.join(entities_type_counter),
            entities_count,
            passages_count,
            passages_char_count,
            passages_char_count / passages_count if passages_count else 0,
            len(splits),
            ', '.join(splits),
        ]

        try:
            csv_writer.writerow(line)
        except UnicodeEncodeError as e:
            print(dataset_name, description)
            csv_writer.writerow([dataset_name, config_name, description, 'error'])
