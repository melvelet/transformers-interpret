import csv
from bigbio.dataloader import BigBioConfigHelpers

conhelps = BigBioConfigHelpers()
dataset_names = [
    'bc5cdr_bigbio_kb',
    'euadr_bigbio_kb',
    'cadec_bigbio_kb',
    'scai_disease_bigbio_kb',
    'ncbi_disease_bigbio_kb',
    'verspoor_2013_bigbio_kb',
]

for dataset_name in dataset_names:
    dataset = conhelps.for_config_name(dataset_name).load_dataset()

    classes = sorted(list(set([e['type'] for split in dataset for document in dataset[split] for e in document['entities']])))
    dataset_entities = []
    dataset_dicts = []

    for split in dataset:
        split_entities = [e['type'] for document in dataset[split] for e in document['entities']]
        dataset_entities += split_entities
        split_dict = {class_: split_entities.count(class_) for class_ in split_entities}
        split_dict['dataset'] = f"{dataset_name}-{split}"
        dataset_dicts.append(split_dict)

    if len(dataset) > 1:
        dataset_dict = {class_: dataset_entities.count(class_) for class_ in dataset_entities}
        dataset_dict['dataset'] = f"{dataset_name}-all"
        dataset_dicts.append(dataset_dict)

    with open(f'entities_{dataset_name}.csv', 'w+', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset'] + classes)
        writer.writeheader()
        writer.writerows(dataset_dicts)
