import json
import csv
from os.path import join
from config import FLAGS
import numpy as np
from data_analysis_utils import mean_fix


def read_to_dict(path):
    path = join(FLAGS.harp_path, path)
    with open(path, 'r') as f:
        res = json.load(f)
    res_dict = dict()
    for item in res:
        res_dict[item["id"]] = item
    return res_dict


def mask_negative(res_dict):
    utils = ['util-LUT','util-DSP','util-FF','util-BRAM']
    for k, v in res_dict.items():
        for target_name, value in v.items():
            if target_name in utils:
                if value < 0:
                    res_dict[k][target_name] = 0


class_res_dict = read_to_dict('src/class.json')
reg_res_dict = read_to_dict('src/regression.json')

data_list = []
mask_negative(reg_res_dict)
# id,valid,perf,util-LUT,util-DSP,util-FF,util-BRAM
with open('../history/submission_9_20.csv', mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    for i, row in enumerate(csv_reader):
        if i == 0:
            data_list.append(row)
            continue
        id = i-1
        alpha = 0.9
        # if class_res_dict[id]['perf'] == 0:
        if row[1] == 'False':
            data_list.append([row[0], False, 0, 0, 0, 0, 0])
        elif float(row[3])>alpha or float(row[4])>alpha or float(row[5])>alpha or float(row[6])>alpha:
            data_list.append([row[0], False, 0, 0, 0, 0, 0])
        else:
            data_list.append([row[0], True, reg_res_dict[id]['perf'], reg_res_dict[id]['util-LUT'], reg_res_dict[id]['util-DSP'], reg_res_dict[id]['util-FF'], reg_res_dict[id]['util-BRAM']])
    
    
with open('output.csv', mode='w', newline='', encoding='utf-8') as file:
    csv_writer = csv.writer(file)
    for row in data_list:
        csv_writer.writerow(row) 

# mean_fix('output.csv')