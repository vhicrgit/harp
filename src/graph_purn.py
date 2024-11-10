import networkx as nx
import os

# flow | 0:控制流 1:data 2:call 3:pragma 4:伪节点

def count_nodes_edges(G):
    print(f'nodes: {len(G.nodes)}')
    print(f'edges: {len(G.edges)}')
    # return len(graph.nodes)
    
def delete_inedges_of_node(G, node_id):
    G.remove_edges_from(list(G.in_edges(node_id)))
    
def delete_outedges_of_node(G, node_id):
    G.remove_edges_from(list(G.out_edges(node_id)))    
    
    
def print_prevs(G, node_id, content='features'):
    prevs = []
    for edge in G.in_edges(node_id):
        text = G.nodes(data=True)[edge[0]][content]
        text = text.replace("{'full_text': [", "").replace("]}", "")
        result = text.strip("'")
        prevs.append(result)
    print(f'prevs: {prevs}')
    
def print_succs(G, node_id, content='features'):
    succs = []
    for edge in G.out_edges(node_id):
        text = G.nodes(data=True)[edge[1]][content]
        text = text.replace("{'full_text': [", "").replace("]}", "")
        result = text.strip("'")
        succs.append(result)
    print(f'succs: {succs}')
    
def get_prevs(G, node_id):
    prevs = []
    for edge in G.in_edges(node_id):
        prevs.append(edge[0])
    return prevs
    
def get_succs(G, node_id):
    succs = []
    for edge in G.out_edges(node_id):
        succs.append(edge[1])
    return succs

def print_inedges(G, node_id):
    for edge in G.in_edges(node_id, data=True):
        print(edge)
        
def print_outedges(G, node_id):
    for edge in G.out_edges(node_id, data=True):
        print(edge)


def is_type(string):
    keywords = ['i32', 'i64', 'double']
    contains_keyword = any(keyword in string for keyword in keywords)
    return contains_keyword

def is_pointer(string):
    keywords = ['i32', 'double']
    contains_keyword = any(keyword in string for keyword in keywords)
    star = '*' in string
    return star and contains_keyword

def remove_alloca(G):
    nodes_to_remove = []
    for node_id, attributes in G.nodes(data=True):
        if attributes['text'] == 'alloca':
            nodes_to_remove.append(node_id)

    for node_id in nodes_to_remove:
        delete_inedges_of_node(G, node_id)
        delete_outedges_of_node(G, node_id)

    G.remove_nodes_from(nodes_to_remove)
    print(f'Removed {len(nodes_to_remove)} nodes with text "alloca"')
    
    
def remove_single_load(G, node_id):
    prevs = get_prevs(G, node_id)
    succs = get_succs(G, node_id)
    
    prev_nodes = dict()
    addr_node = 0
    for prev in prevs:
        text = G.nodes(data=True)[prev]['text']
        if text == 'pseudo_block':
            pass
        elif is_pointer(text):
            addr_node = prev
        else:
            prev_nodes[prev] = G.get_edge_data(prev, node_id)['position'], G.get_edge_data(prev, node_id)['id']
        G.remove_edge(prev, node_id)
    
    succ_nodes = []
    value_node = 0
    value_edge_id = 0
    value_edge_pos = 0
    for succ in succs:
        text = G.nodes(data=True)[succ]['text']
        if text == 'pseudo_block':
            pass
        elif is_type(text):
            value_node = succ
            value_edge_id = G.get_edge_data(node_id, succ)['id']
            value_edge_pos = G.get_edge_data(node_id, succ)['position']
        else:
            succ_nodes.append(succ)
        G.remove_edge(node_id, succ)
    
    # 理论上load的后继要么是下一条指令，要么是load出来的值，下一条指令只能有一个
    # 好吧其实可以有多个，可能会有这种情况 %0 = load ....; i64 %0; add %2, %1, %0
    assert len(succ_nodes) == 1, f"Assertion failed: len(succ_nodes) is {len(succ_nodes)} instead of 1"
    # if len(succ_nodes) != 1:
        
    
    # 直接将addr_node 和 value_node 连起来
    G.add_edge(addr_node, value_node, flow=1, position=value_edge_pos, id=value_edge_id, networkx_key=0)
    # 将load的前驱们和后继连起来
    for prev, data in prev_nodes.items():
        G.add_edge(prev, succ_nodes[0], flow=0, position=data[0], id=data[1], networkx_key=0)
    # 删除load节点
    G.remove_node(node_id)
           

def remove_loads(G):
    loads_to_remove = []
    for node_id, attributes in G.nodes(data=True):
        if node_id == 0:
            # [external] node
            continue
        elif attributes['text'] == 'load':
            loads_to_remove.append(node_id)
            
    for node_id in loads_to_remove:
        remove_single_load(G, node_id)
    
    
def remove_single_store(G, node_id):
    prevs = get_prevs(G, node_id)
    succs = get_succs(G, node_id)
    # store的前驱有3/4个：伪节点，(上一条指令)，store地址，store值
    if len(prevs) > 4:
        print(f"prevs: {prevs}")
    assert len(prevs) <= 4, f"Assertion failed: len(prevs) is {len(prevs)} instead of 4"
    # store的后继驱有2个：伪节点，下一条指令
    assert len(succs) <= 2
    # prev_nodes = dict()
    prev_node = 0
    prev_node_pos = 0
    prev_node_id = 0
    addr_node = 0
    value_node = 0
    value_edge_id = 0
    value_edge_pos = 0    
    for prev in prevs:
        text = G.nodes(data=True)[prev]['text']
        if text == 'pseudo_block':
            pass
        elif is_pointer(text):
            addr_node = prev
        elif is_type(text):
            value_node = prev
            value_edge_id = G.get_edge_data(prev, node_id)['id']
            value_edge_pos = G.get_edge_data(prev, node_id)['position']
        else:
            # 前一条指令
            prev_node = prev
            prev_node_id = G.get_edge_data(prev, node_id)['id']
            prev_node_pos = G.get_edge_data(prev, node_id)['position']
        G.remove_edge(prev, node_id)
    
    succ_node = 0
    succ_node_pos = 0
    succ_node_id = 0
    for succ in succs:
        text = G.nodes(data=True)[succ]['text']
        if text == 'pseudo_block':
            pass
        else:
            # 后一条指令
            succ_node = succ
            succ_node_pos = G.get_edge_data(node_id, succ)['id']
            succ_node_id = G.get_edge_data(node_id, succ)['id']
        G.remove_edge(node_id, succ)

    # 直接将addr_node 和 value_node 连起来
    G.add_edge(value_node, addr_node, flow=1, position=value_edge_pos, id=value_edge_id, networkx_key=0)
    # 将store的前驱和后继连起来
    if len(prevs) >= 4 and len(succs) >= 2:
        G.add_edge(prev_node, succ_node, flow=0, position=prev_node_pos, id=prev_node_id, networkx_key=0)
    # 删除store节点
    G.remove_node(node_id)
           

def remove_stores(G):
    stores_to_remove = []
    for node_id, attributes in G.nodes(data=True):
        if node_id == 0:
            # [external] node
            continue
        elif attributes['text'] == 'store':
            stores_to_remove.append(node_id)
            
    for node_id in stores_to_remove:
        remove_single_store(G, node_id)


def process_all_graphs(path='/home/wqlou/kzw3933/harp/dse_database/data/extend_graphs'):
    for filename in os.listdir(path):
        if filename.endswith('.gexf'):
            graph_path = os.path.join(path, filename)
            print(f"Processing {graph_path}")
            G = nx.read_gexf(graph_path)
            remove_alloca(G)
            remove_loads(G)
            remove_stores(G)
            # 保存修改后的图到新的GEXF文件
            output_gexf_file_path = f"/home/wqlou/kzw3933/harp/dse_database/data/purned_graphs/{filename[:-5]}.gexf"
            nx.write_gexf(G, output_gexf_file_path)
    
# 读取GEXF文件
# gexf_file_path = './3mm/3mm_processed_result.gexf'
# G = nx.read_gexf(gexf_file_path)

# for node_id, attributes in G.nodes(data=True):
#     if attributes['text'] == 'store':
#         # print_prevs(G, node_id)
#         print_prevs(G, node_id, 'text')
#         print_inedges(G, node_id)
#         # print_succs(G, node_id)
#         # print_succs(G, node_id, 'text')
#         # print_outedges(G, node_id)
#         print('===============')


# # 删除属性为 'alloca' 的节点
# nodes_to_remove = []
# for node_id, attributes in G.nodes(data=True):
#     if attributes['text'] == 'alloca':
#         nodes_to_remove.append(node_id)

# # 删除节点及其相关的边
# G.remove_nodes_from(["2",'3'])

# # 保存修改后的图到新的GEXF文件
# count_nodes_edges(G)
# output_gexf_file_path = 'output.gexf'
# nx.write_gexf(G, output_gexf_file_path)

# print(f"Nodes removed: {nodes_to_remove}")
# print(f"Modified graph saved to {output_gexf_file_path}")

# count_nodes_edges(G)
# remove_alloca(G)
# G_undirected = G.to_undirected()
# count_nodes_edges(G)
# remove_loads(G)
# count_nodes_edges(G)
# remove_stores(G)
# count_nodes_edges(G)

# output_gexf_file_path = 'output.gexf'
# nx.write_gexf(G, output_gexf_file_path)

process_all_graphs()