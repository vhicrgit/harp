import networkx as nx
from lxml import etree
import os
import json


def read_all_gexf(file_path):
    all_maps = dict()
    for file in os.listdir(file_path):
        if file.endswith('.gexf'):
            key = file.split('_')[0]
            value = read_gexf_and_create_map(os.path.join(file_path, file))
            all_maps[key] = value
    return all_maps


def read_gexf_and_create_map(file_path):
    G = nx.read_gexf(file_path)
    id_value_map = {}
    
    # 遍历每个节点
    for node in G.nodes(data=True):
        node_id = int(node[0])
        text = node[1].get('text')
        id_value_map[node_id] = text

    return id_value_map



def create_all_idx_text_map(file_path):
    all_maps = dict()
    for file in os.listdir(file_path):
        if file.endswith('.gexf'):
            key = file.split('_')[0]
            value = create_idx_text_map(os.path.join(file_path, file))
            all_maps[key] = value
    return all_maps


def create_idx_text_map(file_path):
    G = nx.read_gexf(file_path)
    nodes = [str(i) for i in range(0, len(G.nodes))]
    sorted_nodes = sorted(nodes)
    labels_idx_map = {node: i for i, node in enumerate(sorted_nodes)}
    
    idx_text_map = {}
    for node in G.nodes(data=True):
        node_id = node[0]
        text = node[1].get('text')
        idx_text_map[labels_idx_map[node_id]] = text
    return idx_text_map




file_path = '/home/wqlou/kzw3933/HARP-MY/dse_database/kaggle/train_data/data/extend_graphs'
fp = '/home/wqlou/kzw3933/HARP-MY/dse_database/kaggle/train_data/data/extend_graphs/2mm_processed_result.gexf'
# read_all_gexf = read_all_gexf(file_path)
# print(read_all_gexf.keys())
# save_path = '/home/wqlou/kzw3933/HARP-MY/save/harp/id2text.json'
# with open(save_path, 'w') as file:
#     json.dump(read_all_gexf, file, indent=4)

all_maps = create_all_idx_text_map(file_path)
print(all_maps.keys())
save_path = '/home/wqlou/kzw3933/HARP-MY/save/harp/idx2text.json'
with open(save_path, 'w') as file:
    json.dump(all_maps, file, indent=4)