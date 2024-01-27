import json 
import os


# custom setting 
source_dir = '../data/medical/ufo/train.json'
target_dir = '../dataaug/ufo1.json'
output_dir = '../data/medical/ufo/merged.json'
iter_limit = 10


# load annotation files
with open(source_dir) as f:
    ori_data = json.load(f)
    print(f'{len(ori_data["images"].keys())} source data are loaded.')
with open(target_dir) as f:
    aug_data = json.load(f)
    print(f'{len(aug_data["images"].keys())} target data are loaded.')

# merge annotations with iter_limit
for i, key in enumerate(aug_data['images'].keys()):
    if i == iter_limit: break
    ori_data['images'].setdefault(key, {})
    ori_data['images'][key] = aug_data['images'][key]

# save merged_annotation file 
with open(output_dir, 'w') as f:
    json.dump(ori_data, f)
    print(f'{len(ori_data["images"].keys())} output data are saved.')
