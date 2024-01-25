import os
import json
import shutil
import numpy as np
from tqdm import tqdm


seed = 2
np.random.seed(seed)

with open('../data/medical/ufo/train.json') as f:
    data = json.load(f)
    
key_images = list(data['images'].keys())
fold_random = np.random.permutation(len(key_images))

path_img = '../data/medical/img/'
path_json = '../data/medical/ufo/'
K = 5
split_n = len(key_images)//K

for idx_k in tqdm(range(K)):
    data_train = {'images': dict()}
    data_valid = {'images': dict()}

    for idx in range(len(key_images)):
        image_name = key_images[fold_random[idx]]

        # make folder
        if not os.path.exists(path_img + str(idx_k) + '_train'):
            os.mkdir(path_img + str(idx_k) + '_train')
        if not os.path.exists(path_img + str(idx_k) + '_valid'):
            os.mkdir(path_img + str(idx_k) + '_valid')
        

        # json data
        if idx_k * split_n <= idx < (idx_k + 1) * split_n:    
            data_valid['images'][image_name] = data['images'][image_name]
            shutil.copyfile(path_img + 'train/' + image_name, path_img + str(idx_k) + '_valid/' + image_name)
        else:
            data_train['images'][image_name] = data['images'][image_name]
            shutil.copyfile(path_img + 'train/' + image_name, path_img + str(idx_k) + '_train/' + image_name)


    # train
    with open(os.path.join(path_json, str(idx_k) + '_train.json'), 'w') as f:
        json.dump(data_train, f, indent=4, ensure_ascii = False)
    
    # valid
    with open(os.path.join(path_json, str(idx_k) + '_valid.json'), 'w') as f:
        json.dump(data_valid, f, indent=4, ensure_ascii = False)

