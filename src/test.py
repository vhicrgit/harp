import torch_geometric
import torch
import json
from torch_geometric.data import Data



data = torch.load('/home/wqlou/kzw3933/HARP/save/harp/v18_MLP-True-extended-pseudo-block-connected-hierarchy-class_edge-position-True_norm_with-invalid_False-normalization_speedup-log2_tag_whole-machsuite-poly_perfutil-DSPutil-BRAMutil-LUTutil-FF/data_10.pt')

# # 检查加载的内容是否是 PyTorch Geometric 的 Data 对象
# if isinstance(data, Data):
#     # 创建一个字典来存储 Data 对象的所有属性
#     data_dict = {}
    
#     # 将 Data 对象的各个属性转换为 JSON 兼容的格式
#     for key, value in data:
#         if isinstance(value, torch.Tensor):
#             data_dict[key] = value.tolist()  # 将张量转换为列表
#         elif isinstance(value, (list, dict)):
#             data_dict[key] = value  # 列表或字典直接保存
#         else:
#             data_dict[key] = str(value)  # 其他类型（如 int、float）转为字符串
    
#     # 写入 JSON 文件
#     json_file_path = 'data_output.json'
#     with open(json_file_path, 'w') as json_file:
#         json.dump(data_dict, json_file, indent=4)
    
#     print(f"Data has been written to {json_file_path}")
# else:
#     print("The loaded file does not contain a 'Data' object.")
print(data.kernel_name)
print(data.edge_index)