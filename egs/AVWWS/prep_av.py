# -*- coding: utf-8 -*-
# @Time    : 6/23/21 3:19 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : prep_sc.py

import numpy as np
import json
import os

label_set = np.loadtxt('./data/AVWWS_class_labels_indices.csv', delimiter=',', dtype='str')
label_map = {}
for i in range(1, len(label_set)):
    label_map[label_set[i][1]] = label_set[i][0]
print(label_map)

# generate  json files
if os.path.exists('./data/datafiles') == False:
    os.mkdir('./data/datafiles')
    base_path = './data/AVWWS/'
    for split in ['train', 'dev', 'eval']:
        wav_list = []
        ne_po = ['negative', 'positive']
        for tmp in ne_po: 
            dis_audio = ['far', 'middle', 'near']
            
            for dis in dis_audio:
                filelist = os.listdir(base_path + tmp + '/audio/' + split + '/' + dis)
                for file_name in filelist:
                    cur_label = '0' if tmp == 'negative' else '1'
                    cur_path = os.path.abspath(base_path + tmp + '/audio/' + split + '/' + dis + '/' + file_name)
                    cur_dict = {"wav": cur_path, "labels": '/m/av' + cur_label.zfill(2)}
                    wav_list.append(cur_dict)

        if split == 'train':
            with open('./data/datafiles/AVWWS_train_data.json', 'w') as f:
                json.dump({'data': wav_list}, f, indent=1)
        if split == 'dev':
            with open('./data/datafiles/AVWWS_dev_data.json', 'w') as f:
                json.dump({'data': wav_list}, f, indent=1)
        if split == 'eval':
            with open('./data/datafiles/AVWWS_eval_data.json', 'w') as f:
                json.dump({'data': wav_list}, f, indent=1)
        print(split + ' data processing finished, total {:d} samples'.format(len(wav_list)))

    print('AVWWS dataset processing finished.')

