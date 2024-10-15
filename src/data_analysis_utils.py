import json
import os
import numpy as np

import pandas as pd
import re

def create_id_kernel_map():
    file_path = './test.csv'
    df = pd.read_csv(file_path)
    # 提取 'id' 和 'kernel_name'，kernel_name 是 '__kernel__' 后的部分
    def extract_kernel_name(kernel_string):
        match = re.search(r'__kernel__-([\w\-]+)', kernel_string)
        if match:
            return match.group(1)
        return None
    # 创建从 'id' 到 'kernel_name' 的映射
    id_kernel_map = {}
    for idx, row in df.iterrows():
        kernel_name = extract_kernel_name(row['designs'])
        if kernel_name:
            id_kernel_map[row['id']] = kernel_name

    # print(id_kernel_map)
    return id_kernel_map

def use_mean_target(id_kernel_map, mead_map, path='./fm_v4.csv'):
    df = pd.read_csv(path)
    for index, row in df.iterrows():
        if row['valid'] == True:
            kernel = id_kernel_map[index]
            df.at[index, 'perf'] = 1e7 / np.exp(mead_map[kernel]['perf'])
            df.at[index, 'util-LUT'] = mead_map[kernel]['util-LUT']
            df.at[index, 'util-FF'] = mead_map[kernel]['util-FF']
            df.at[index, 'util-BRAM'] = mead_map[kernel]['util-BRAM']
            df.at[index, 'util-DSP'] = mead_map[kernel]['util-DSP']
    df.to_csv(path, index=False)

# 统计v21下每个kernel有多少个design（按照valid=True,Valid=False各多少个）
def designs_cnt(data):
    cnt = dict()
    all_cnt = dict()
    all_cnt['all'] = 0
    all_cnt['valid'] = 0
    all_cnt['invalid'] = 0
    for kernel_name, kernel in data.items():
        cnt[kernel_name] = designs_cnt_single(kernel)
        # for _, data in cnt.items():
        all_cnt['all'] += cnt[kernel_name]['all']
        all_cnt['valid'] += cnt[kernel_name]['valid']
        all_cnt['invalid'] += cnt[kernel_name]['invalid']
        
    print(f"Number of Designs: {all_cnt['all']} (True: {all_cnt['valid']}, False: {all_cnt['invalid']}, ratio: {all_cnt['valid']/all_cnt['invalid']})")

def designs_cnt_single(data):
    cnt = dict()
    valid_cnt = 0
    invalid_cnt = 0
    for _, design in data.items():
        if design['valid'] == True:
            valid_cnt += 1
        else:
            invalid_cnt += 1
    cnt['all'] = valid_cnt + invalid_cnt
    cnt['valid'] = valid_cnt
    cnt['invalid'] = invalid_cnt
    return cnt

def target_cnt(all_data):
    all_cnt = {
        'perf': [],
        'util-LUT': [],
        'util-DSP': [],
        'util-FF': [],
        'util-BRAM': []
    }
    cnt = dict()
    for kernel_name, kernel in all_data.items():
        cnt[kernel_name] = target_cnt_single(kernel)
        # for _, data in cnt.items():
        for key in all_cnt:
            all_cnt[key].extend(cnt[kernel_name][key])
    for key, values in all_cnt.items():
        print(f"{key}: {np.mean(values)}")
    return all_cnt

def target_cnt_single(data):
    cnt = {
        'perf': [],
        'util-LUT': [],
        'util-DSP': [],
        'util-FF': [],
        'util-BRAM': []
    }
    for _, design in data.items():
        if design['valid'] == False:
            cnt['perf'].append(np.log(1e7 / (4000+1e-6)))
            for target_name, value in design['res_util'].items():
                if 'util' in target_name:
                    cnt[target_name].append(5e-3)
        else:
            cnt['perf'].append(np.log(1e7 / (float(design['perf'])+1e-6)))
            for target_name, value in design['res_util'].items():
                if 'util' in target_name:
                    cnt[target_name].append(float(value))
    return cnt

def get_json_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def get_all_json_data_in_dir(path):
    all_data = dict()
    for filename in os.listdir(path):
        if filename.endswith('.json'):
            basename = filename.split('.')[0]
            all_data[basename] = get_json_data(os.path.join(path, filename))
    return all_data

def lut_analysis(data):
    count_dict = {f"[{i/10:.1f},{(i+1)/10:.1f})": 0 for i in range(10)}

    for value in data:
        for i in range(10):
            if i / 10 <= value < (i + 1) / 10:
                count_dict[f"[{i/10:.1f},{(i+1)/10:.1f})"] += 1
                break 
    for interval, count in count_dict.items():
        print(f"{interval}: {count}")
        
def calc_mean_per_kernel(all_data):
    mean_dict = dict()
    for kernel_name, data in all_data.items():
        mean_dict[kernel_name] = calc_mean_signle_kernel(data)    
    return mean_dict

def calc_mean_signle_kernel(data):
    mean = {
        'perf': 0,
        'util-LUT': 0,
        'util-DSP': 0,
        'util-FF': 0,
        'util-BRAM': 0        
    }
    cnt = target_cnt_single(data)
    for key, value in cnt.items():
        mean[key] = np.mean(value)
    return mean

def mask_big_util():
    df = pd.read_csv('D:\WorkSpace\kaggle\data_analysis\submission (5).csv')

    def mask_false_row(row):
        alpha = 0.8
        if row['util-LUT'] > alpha or row['util-DSP'] > alpha or row['util-FF'] > alpha or row['util-BRAM'] > alpha:
            row['valid'] = False
            row['perf'] = 0
            row['util-LUT'] = 0
            row['util-DSP'] = 0
            row['util-FF'] = 0
            row['util-BRAM'] = 0
        return row
    def mask_true_row(row):
        if row['valid'] is True:
            row['perf'] = 999999
            row['util-LUT'] = 0
            row['util-DSP'] = 0
            row['util-FF'] = 0
            row['util-BRAM'] = 0
        return row

    df = df.apply(mask_false_row, axis=1)
    df = df.apply(mask_true_row, axis=1)
    df.to_csv('modified_file.csv', index=False)
    print("操作完成，结果已保存到 'modified_file.csv'")

# 将分类和回归的结果合并(csv文件)
def combine_class_reg_csv(class_path='./mask0.9_v21mean.csv', 
                          regression_path='./harp_regression_10.3.csv', 
                          output_path='./mask0.9_combined_harp_util.csv'):
    df_gnndse = pd.read_csv(regression_path)
    df_harp = pd.read_csv(class_path)

    class_item = ['id','valid', 'perf']
    regression_item = ['id', 'util-LUT', 'util-DSP', 'util-FF', 'util-BRAM']

    df_combined = pd.merge(df_harp[class_item], df_gnndse[regression_item], on='id', how='inner') 
    def mask_row(row):
        alpha = 0.8
        if row['valid'] is False:
            row['perf'] = 0
            row['util-LUT'] = 0
            row['util-DSP'] = 0
            row['util-FF'] = 0
            row['util-BRAM'] = 0
        return row
    df_combined = df_combined.apply(mask_row, axis=1)
    df_combined.to_csv(output_path, index=False)
        

# v18_all_data = get_all_json_data_in_dir('designs/v18')
# v20_all_data = get_all_json_data_in_dir('designs/v20')
v21_all_data = get_all_json_data_in_dir('designs/v21')
# designs_cnt(v18_all_data)
# designs_cnt(v20_all_data)
# designs_cnt(v21_all_data)
# target_cnt(v18_all_data)
# target_cnt(v20_all_data)
# res = target_cnt(v21_all_data)

# print(1e7 / np.exp(3.9761188363287503))
# lut_analysis(res['util-LUT'])
# print(np.mean([1,1.2,3,9.4]))
v21_mean = calc_mean_per_kernel(v21_all_data)
# v18_mean = calc_mean_per_kernel(v18_all_data)
# print("v21_mean:", v21_mean)
# print("v18_mean:", v18_mean)

# v21_all_data = get_all_json_data_in_dir('designs/v21')
# v21_mean = calc_mean_per_kernel(v21_all_data)
id_kernel_map = create_id_kernel_map()
use_mean_target(id_kernel_map, v21_mean)

# mask_big_util()