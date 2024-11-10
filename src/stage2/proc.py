from result import Result
import pickle
import json

kernels = ['3mm', 'atax-medium', 'covariance', 'fdtd-2d', 'gemm-p', \
         'gemver-medium', 'jacobi-2d', 'symm-opt', 'trmm-opt', 'syr2k']


# print(list(data_dict.values())[0])

submit = []
for j, name in enumerate(kernels):
    with open(f'./pickle/{name}.pickle', 'rb') as f:
        data = pickle.load(f)
    data_dict = dict()
    for k, v in data.items():
        perf = k[0]
        config = k[1]
        data_dict[perf] = config
        
    # top 5 designs
    data_dict = dict(sorted(data_dict.items())[:5])
    for k in range(5):
        sample = list(data_dict.values())[k]
        sample = sample.split('.')
        sample_dict = dict()
        sample_dict['kernel'] = name
        sample_dict['point'] = dict()
        for i in range(len(sample)):
            tmp = sample[i].split('-')
            if "tensor" not in tmp[1]:
                sample_dict['point'][tmp[0]] = tmp[1]
            else:
                num = tmp[1].split('[')[1].split(']')[0]
                sample_dict['point'][tmp[0]] = num

        submit.append(sample_dict)
print(submit)
with open('./submit.json', 'w') as f:
    json.dump(submit, f, indent=4)
# with open('./json/3mm.json', 'w') as f:
#     json.dump(data, f, indent=4)