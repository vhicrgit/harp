import json
import csv


def mask_negative(res_dict):
    utils = ['util-LUT','util-DSP','util-FF','util-BRAM']
    for k, v in res_dict.items():
        for target_name, value in v.items():
            if target_name in utils:
                if value < 0:
                    res_dict[k][target_name] = 0


path = "/root/autodl-tmp/kaggle/src/result.json"
outpath = "/root/autodl-tmp/kaggle/src/result_dict.json"
with open(path, 'r') as f:
    res = json.load(f)
res_dict = dict()
for item in res:
    res_dict[item["id"]] = item
with open(outpath, 'w') as f:
    json.dump(res_dict, f, indent=4)
    
data_list = []
mask_negative(res_dict)
# id,valid,perf,util-LUT,util-DSP,util-FF,util-BRAM
with open('submission.csv', mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    for i, row in enumerate(csv_reader):
        if i == 0:
            data_list.append(row)
            continue
        id = i-1
        if row[1] == "True":
            # data_list.append([row[0], row[1], res_dict[id]['perf'], res_dict[id]['util-LUT'], res_dict[id]['util-DSP'], res_dict[id]['util-FF'], res_dict[id]['util-BRAM']])
            data_list.append([row[0], row[1], 6000, 0, 0, 0, 0])
        else:
            data_list.append([row[0], row[1], 0, 0, 0, 0, 0])
    
    
with open('output.csv', mode='w', newline='', encoding='utf-8') as file:
    csv_writer = csv.writer(file)
    for row in data_list:
        csv_writer.writerow(row) 