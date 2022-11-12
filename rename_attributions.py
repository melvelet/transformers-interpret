import os
from pathlib import Path

base_dir = 'results\\attributions\\'
directory = os.fsencode(base_dir)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith("_attribution_stats.json"):
        base_file_name = filename.replace('_attribution_stats.json', '')
        new_file_name = '_'.join([i for i in base_file_name.split('_') if not i[0].isdigit()])
        for file_type in ['_attributed_entities.json', '_attribution_stats.json']:
            path = Path(f"{base_dir}{base_file_name}{file_type}")
            if path.is_file():
                os.rename(f"{base_dir}{base_file_name}{file_type}", f"{base_dir}{new_file_name}{file_type}")
            elif 'stats' in file_type:
                print(f'can\'t find {path}')
