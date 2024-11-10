import json
import pandas as pd
import networkx as nx
from os.path import join
import os

# # 读取 CSV 文件
# file_path = '/home/wqlou/kzw3933/harp/src/output.csv'
# df = pd.read_csv(file_path)

# # 检查 id 列是否有重复值
# duplicate_ids = df[df.duplicated(subset=['id'], keep=False)]

# if not duplicate_ids.empty:
#     print("存在重复的 id:")
#     print(duplicate_ids)
# else:
#     print("没有发现重复的 id")


harp_path = '/home/wqlou/kzw3933/harp'
labels_map_dir = join(harp_path, 'save/harp/idx2text_origin.json')
graph_dir_path = '/home/wqlou/kzw3933/harp/dse_database/data/extend_graphs'


labels_map = dict()
for filename in os.listdir(graph_dir_path):
    basename = filename.split('_')[0]
    path = join(graph_dir_path, filename)
    G = nx.read_gexf(path)
    labels_map_ = dict()
    for i, node in enumerate(G.nodes(data=True)):
        if i==0:
            labels_map_[i] = node[1]['text']
        else:
            labels_map_[i] = node[1]['full_text']
    labels_map[basename] = labels_map_

print("Nodes:", labels_map)
with open(labels_map_dir, 'w') as f:
    json.dump(labels_map, f, indent=4)

# path = '/home/wqlou/kzw3933/harp/dse_database/data/purned_graphs/trmm_processed_result.gexf'
# G = nx.read_gexf(path)
# print(G.nodes(data=True))