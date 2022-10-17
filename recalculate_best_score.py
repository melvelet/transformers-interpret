import json
import os
from statistics import StatisticsError, median, stdev, variance, mean

base_dir = 'results/scores/'
directory = os.fsencode(base_dir)
k_values = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20]


def _calculate_statistical_function(attr: str, func: str = None, take_best_rationale: bool = False,
                                    take_best_rationale_threshold: float = 0.05):
    if func == 'median':
        func = median
    elif func == 'stdev':
        func = stdev
    elif func == 'variance':
        func = variance
    else:
        func = mean

    try:
        if take_best_rationale:
            mode = 'top_k'
            best_rationale_scores = []
            best_rationale_k_values = []
            for e in raw_scores:
                best_rationale_compdiff = -1
                best_rationale_per_mode_k_value = 0
                best_rationale_compdiff_prev = 0
                prev_k = k_values[-1]
                for k in reversed(k_values):
                    if best_rationale_compdiff_prev - e['compdiff'][mode][str(k)] > take_best_rationale_threshold:
                        best_rationale_compdiff = e[attr][mode][str(prev_k)]
                        best_rationale_per_mode_k_value = prev_k
                        break
                    best_rationale_compdiff_prev = e['compdiff'][mode][str(k)]
                    prev_k = k
                if best_rationale_compdiff == -1:
                    best_rationale_compdiff = e[attr][mode]['2']
                    best_rationale_per_mode_k_value = 2
                best_rationale_scores.append(best_rationale_compdiff)
                best_rationale_k_values.append(best_rationale_per_mode_k_value)
            return func(best_rationale_scores), func(best_rationale_k_values)

    except StatisticsError:
        print(f"Can't calculate {func.__name__} for attribute {attr}. Too few data points...")
        return None


for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith("_raw_entities.json"):
        base_file_name = filename.replace('_raw_entities.json', '')
        new_file_name = '_'.join([i for i in base_file_name.split('_') if not i[0].isdigit()])
        for file_type in ['_raw_entities.json', '_raw_scores.json', '_scores.json']:
            os.rename(f"{base_dir}{base_file_name}{file_type}", f"{base_dir}{new_file_name}{file_type}")
        with open(f"{base_dir}{new_file_name}_raw_scores.json", 'r') as f:
            raw_scores = json.load(f)
        with open(f"{base_dir}{new_file_name}_scores.json", 'r') as f:
            scores = json.load(f)

        for func in ['mean', 'median']:
            for threshold in [5, 10, 15, 20]:
                best_rationale_threshold = 0.01 * threshold
                for measure in ['comprehensiveness', 'sufficiency', 'compdiff']:
                    scores['scores'][func][f'{measure}_Best_{threshold}'] = _calculate_statistical_function(measure,
                                                                                                            take_best_rationale=True,
                                                                                                            take_best_rationale_threshold=best_rationale_threshold)

        with open(f"{base_dir}{new_file_name}_scores.json", 'w+') as f:
            print(type(raw_scores))
            json.dump(scores, f)
