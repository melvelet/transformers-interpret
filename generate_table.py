import csv
import json
from pathlib import Path

import latextable
from texttable import Texttable
from tabulate import tabulate

attribution_types = [
    # 'LIG',
    'LGXA',
    # 'GradCAM',
]

dataset_names = [
    'bc5cdr',
    'ncbi_disease',
    'ddi_corpus',
    'cadec',
]

dataset_names_table = [
    '1',  # BC5CDR',
    '2',  # 'NCBI',
    '3',  # 'DDI',
    # '4-1',  # 'CADEC-Dis',
    '4-2',  # 'CADEC-Dru',
]

huggingface_models = [
    'bioelectra-discriminator',
    'roberta',
]

huggingface_models_pretty = {
    'bioelectra-discriminator': 'BioElectra',
    'roberta': 'RoBERTa',
}

measure_short = {
    'comprehensiveness': 'C',
    'sufficiency': 'S',
    'compdiff': 'CS',
}

include_options = ['include', 'exclude']

measures = [
    'comprehensiveness',
    'sufficiency',
    'compdiff',
]

modes = [
    'top_k',
    'continuous',
]


def combine_json_files():
    result = {dataset:
                  {model: {
                      attr.lower(): {
                          entity: {
                              include: {}
                              for include in ['include', 'exclude']
                          } for entity in ['disease', 'drug']
                      } for attr in attribution_types
                  } for model in huggingface_models}
              for dataset in dataset_names}

    for dataset in dataset_names:
        for model in huggingface_models:
            for attr in attribution_types:
                for entity in ['disease', 'drug']:
                    for include in ['include', 'exclude']:
                        path = Path(f"./results/scores/{dataset}_{entity}_{model}_{attr.lower()}_{include}_scores.json")
                        if path.is_file():
                            with open(path, 'r') as f:
                                data = json.load(f)
                                result[dataset][model][attr.lower()][entity][include] = data

    with open("./results/scores/all_scores.json", 'w+') as f:
        json.dump(result, f)


def load_data():
    path = Path("./results/scores/all_scores.json")
    if not path.is_file():
        combine_json_files()
    with open(path, 'r') as f:
        result = json.load(f)
    return result


def save_table(table_data, file_name, fieldnames):
    with open(f"./results/tables/{file_name}.csv", 'w+', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in table_data:
            writer.writerow(row)

    # with open(f"./results/tables/{file_name}.txt", 'w+', newline='') as f:
    #     table = Texttable()
    #     table.set_cols_align(["c"] * 4)
    #     table.set_deco(Texttable.HEADER | Texttable.VLINES)
    #     tabulate(table_data, headers='keys', tablefmt='latex')
    #     latex = latextable.draw_latex(table, caption="A comparison of rocket features.")
    #     f.write(latex)


def create_include_exclude_table(data):
    for k in [2, 5, 10, 20]:
        table_data = []
        for model in huggingface_models:
            for attr in attribution_types:
                for include in include_options:
                    bc5 = data['bc5cdr'][model][attr.lower()]['disease'][include]
                    ncbi = data['ncbi_disease'][model][attr.lower()]['disease'][include]
                    ddi = data['ddi_corpus'][model][attr.lower()]['drug'][include]
                    cadec_dis = data['cadec'][model][attr.lower()]['disease'][include]
                    cadec_dru = data['cadec'][model][attr.lower()]['drug'][include]
                    raw_data = [bc5, ncbi, ddi, cadec_dis, cadec_dru]
                    row_data = {'Config': f"{huggingface_models_pretty[model]}: {attr} ({include})"}
                    for dataset, d in zip(dataset_names_table, raw_data):
                        for measure in measures:
                            if d:
                                row_data[f"{measure_short[measure]} ({dataset})"] =\
                                    round(d['scores']['mean'][measure]['top_k'][str(k)], 2)

                    table_data.append(row_data)

        fieldnames = ['Config'] + [f"{measure_short[measure]} ({dataset})"
                                   for dataset in dataset_names_table
                                   for measure in measures]
        save_table(table_data, f'include_exclude_table_{k}', fieldnames)


def create_contiguous_topk_table(data):
    # for k in [2, 5, 10, 20]:
    table_data = []
    for model in huggingface_models:
        for attr in attribution_types:
            for k in [10, 20]:
                bc5 = data['bc5cdr'][model][attr.lower()]['disease']['include']
                ncbi = data['ncbi_disease'][model][attr.lower()]['disease']['include']
                ddi = data['ddi_corpus'][model][attr.lower()]['drug']['include']
                # cadec_dis = data['cadec'][model][attr.lower()]['disease'][include]
                cadec_dru = data['cadec'][model][attr.lower()]['drug']['include']
                raw_data = [bc5, ncbi, ddi, cadec_dru]
                for mode in modes:
                    row_data = {'Config': f"{huggingface_models_pretty[model]}: {attr} ({mode[0:3]}-k={k})"}
                    for dataset, d in zip(dataset_names_table, raw_data):
                        for measure in measures:
                            if d:
                                row_data[f"{measure_short[measure]} ({dataset})"] = \
                                    round(d['scores']['mean'][measure][mode][str(k)], 2)

                    table_data.append(row_data)

    fieldnames = ['Config'] + [f"{measure_short[measure]} ({dataset})"
                               for dataset in dataset_names_table
                               for measure in measures]
    save_table(table_data, f'contiguous_topk_table_10_20', fieldnames)


def create_bottom_k_table(data):
    # for k in [2, 5, 10, 20]:
    table_data = []
    for model in huggingface_models:
        for attr in attribution_types:
            for k in [2, 5, 10]:
                bc5 = data['bc5cdr'][model][attr.lower()]['disease']['include']
                ncbi = data['ncbi_disease'][model][attr.lower()]['disease']['include']
                ddi = data['ddi_corpus'][model][attr.lower()]['drug']['include']
                # cadec_dis = data['cadec'][model][attr.lower()]['disease'][include]
                cadec_dru = data['cadec'][model][attr.lower()]['drug']['include']
                raw_data = [bc5, ncbi, ddi, cadec_dru]
                row_data = {'Config': f"{huggingface_models_pretty[model]}: {attr} (bottom-k={k})"}
                for dataset, d in zip(dataset_names_table, raw_data):
                    for measure in measures:
                        if d:
                            row_data[f"{measure_short[measure]} ({dataset})"] = \
                                round(d['scores']['mean'][measure]['bottom_k'][str(k)], 2)

                table_data.append(row_data)

    fieldnames = ['Config'] + [f"{measure_short[measure]} ({dataset})"
                               for dataset in dataset_names_table
                               for measure in measures]
    save_table(table_data, f'bottom_k_table', fieldnames)


def create_overview_table(data):
    table_data = []
    for model in huggingface_models:
        for k in [2, 5, 10, 20]:
            for attr in attribution_types:
                bc5 = data['bc5cdr'][model][attr.lower()]['disease']['include']
                ncbi = data['ncbi_disease'][model][attr.lower()]['disease']['include']
                ddi = data['ddi_corpus'][model][attr.lower()]['drug']['include']
                # cadec_dis = data['cadec'][model][attr.lower()]['disease'][include]
                cadec_dru = data['cadec'][model][attr.lower()]['drug']['include']
                raw_data = [bc5, ncbi, ddi, cadec_dru]
                for mode in modes:
                    row_data = {'Config': f"{huggingface_models_pretty[model]}: {attr} ({mode[0:3]}-k={k})"}
                    for dataset, d in zip(dataset_names_table, raw_data):
                        for measure in measures:
                            if d:
                                row_data[f"{measure_short[measure]} ({dataset})"] = \
                                    round(d['scores']['mean'][measure][mode][str(k)], 2)

                    table_data.append(row_data)

                # row_data = {'Config': f"{huggingface_models_pretty[model]}: {attr} (score(Best))"}
                # for dataset, d in zip(dataset_names_table, raw_data):
                #     for measure in measures:
                #         if d:
                #             row_data[f"{measure_short[measure]} ({dataset})"] = \
                #                 round(d['scores']['mean'][f"{measure}_Best_5"][0], 2)
                # table_data.append(row_data)
                #
                # row_data = {'Config': f"{huggingface_models_pretty[model]}: {attr} (k-value(Best))"}
                # for dataset, d in zip(dataset_names_table, raw_data):
                #     for measure in measures:
                #         if d:
                #             row_data[f"{measure_short[measure]} ({dataset})"] = \
                #                 round(d['scores']['mean'][f"{measure}_Best_5"][1], 2)
                # table_data.append(row_data)

    fieldnames = ['Config'] + [f"{measure_short[measure]} ({dataset})"
                               for dataset in dataset_names_table
                               for measure in measures]
    save_table(table_data, f'overview_table_lgxa', fieldnames)


def create_timing_table(data):
    table_data = []
    for model in huggingface_models:
        for k in [2, 5, 10, 20]:
            for attr in attribution_types:
                bc5 = data['bc5cdr'][model][attr.lower()]['disease']['include']
                ncbi = data['ncbi_disease'][model][attr.lower()]['disease']['include']
                ddi = data['ddi_corpus'][model][attr.lower()]['drug']['include']
                # cadec_dis = data['cadec'][model][attr.lower()]['disease'][include]
                cadec_dru = data['cadec'][model][attr.lower()]['drug']['include']
                raw_data = [bc5, ncbi, ddi, cadec_dru]
                for mode in modes:
                    row_data = {'Config': f"{huggingface_models_pretty[model]}: {attr} ({mode[0:3]}-k={k})"}
                    for dataset, d in zip(dataset_names_table, raw_data):
                        for measure in measures:
                            if d:
                                row_data[f"{measure_short[measure]} ({dataset})"] = \
                                    round(d['scores']['mean'][measure][mode][str(k)], 2)

                    table_data.append(row_data)

    fieldnames = ['Config'] + [f"{measure_short[measure]} ({dataset})"
                               for dataset in dataset_names_table
                               for measure in measures]
    save_table(table_data, f'overview_table_lgxa', fieldnames)


if __name__ == '__main__':
    data = load_data()
    create_overview_table(data)
