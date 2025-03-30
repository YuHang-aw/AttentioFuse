import os
import re
import pandas as pd
import re
import networkx as nx
import pandas as pd
import itertools
import numpy as np
from os.path import join
from copy import deepcopy
# from config_path import REACTOM_PATHWAY_PATH
# from data.gmt_reader import GMT

reactome_base_dir = './'
relations_file_name = 'ReactomePathwaysRelation.txt'
pathway_names = 'ReactomePathways.txt'
pathway_genes = 'ReactomePathways.gmt'


# data_dir = os.path.dirname(__file__)
class GMT():
    # genes_cols : start reading genes from genes_col(default 1, it can be 2 e.g. if an information col is added after the pathway col)
    # pathway col is considered to be the first column (0)
    def load_data(self, filename, genes_col=1, pathway_col=0):

        data_dict_list = []
        with open(filename) as gmt:

            data_list = gmt.readlines()

            # print data_list[0]
            for row in data_list:
                genes = row.strip().split('\t')
                genes = [re.sub('_copy.*', '', g) for g in genes]
                genes = [re.sub('\\n.*', '', g) for g in genes]
                for gene in genes[genes_col:]:
                    pathway = genes[pathway_col]
                    dict = {'group': pathway, 'gene': gene}
                    data_dict_list.append(dict)

        df = pd.DataFrame(data_dict_list)
        # print df.head()

        return df

    def load_data_dict(self, filename):

        data_dict_list = []
        dict = {}
        with open(os.path.join(data_dir, filename)) as gmt:
            data_list = gmt.readlines()

            # print data_list[0]
            for row in data_list:
                genes = row.split('\t')
                dict[genes[0]] = genes[2:]

        return dict

    def write_dict_to_file(self, dict, filename):
        lines = []
        with open(filename, 'w') as gmt:
            for k in dict:
                str1 = '	'.join(str(e) for e in dict[k])
                line = str(k) + '	' + str1 + '\n'
                lines.append(line)
            gmt.writelines(lines)
        return

    def __init__(self):

        return





def add_edges(G, node, n_levels):
    edges = []
    source = node
    for l in range(n_levels):
        target = node + '_copy' + str(l + 1)
        edge = (source, target)
        source = target
        edges.append(edge)

    G.add_edges_from(edges)
    return G


def complete_network(G, n_leveles=4):
    sub_graph = nx.ego_graph(G, 'root', radius=n_leveles)
    terminal_nodes = [n for n, d in sub_graph.out_degree() if d == 0]
    distances = [len(nx.shortest_path(G, source='root', target=node)) for node in terminal_nodes]
    for node in terminal_nodes:
        distance = len(nx.shortest_path(sub_graph, source='root', target=node))
        if distance <= n_leveles:
            diff = n_leveles - distance + 1
            sub_graph = add_edges(sub_graph, node, diff)

    return sub_graph


def get_nodes_at_level(net, distance):
    # get all nodes within distance around the query node
    nodes = set(nx.ego_graph(net, 'root', radius=distance))

    # remove nodes that are not **at** the specified distance but closer
    if distance >= 1.:
        nodes -= set(nx.ego_graph(net, 'root', radius=distance - 1))

    return list(nodes)


def get_layers_from_net(net, n_levels):
    layers = []
    for i in range(n_levels):
        nodes = get_nodes_at_level(net, i)
        dict = {}
        for n in nodes:
            n_name = re.sub('_copy.*', '', n)
            next = net.successors(n)
            dict[n_name] = [re.sub('_copy.*', '', nex) for nex in next]
        layers.append(dict)
    return layers


class Reactome():

    def __init__(self):
        self.pathway_names = self.load_names()
        self.hierarchy = self.load_hierarchy()
        self.pathway_genes = self.load_genes()

    def load_names(self):
        filename = join(reactome_base_dir, pathway_names)
        df = pd.read_csv(filename, sep='\t')
        df.columns = ['reactome_id', 'pathway_name', 'species']
        return df

    def load_genes(self):
        filename = join(reactome_base_dir, pathway_genes)
        gmt = GMT()
        df = gmt.load_data(filename, pathway_col=1, genes_col=3)
        return df

    def load_hierarchy(self):
        filename = join(reactome_base_dir, relations_file_name)
        df = pd.read_csv(filename, sep='\t')
        df.columns = ['child', 'parent']
        return df


class ReactomeNetwork():

    def __init__(self):
        self.reactome = Reactome()  # low level access to reactome pathways and genes
        self.netx = self.get_reactome_networkx()

    def get_terminals(self):
        terminal_nodes = [n for n, d in self.netx.out_degree() if d == 0]
        return terminal_nodes

    def get_roots(self):

        roots = get_nodes_at_level(self.netx, distance=1)
        return roots

    # get a DiGraph representation of the Reactome hierarchy
    def get_reactome_networkx(self):
        if hasattr(self, 'netx'):
            return self.netx
        hierarchy = self.reactome.hierarchy
        # filter hierarchy to have human pathways only
        human_hierarchy = hierarchy[hierarchy['child'].str.contains('HSA')]
        net = nx.from_pandas_edgelist(human_hierarchy, 'child', 'parent', create_using=nx.DiGraph())
        net.name = 'reactome'

        # add root node
        roots = [n for n, d in net.in_degree() if d == 0]
        root_node = 'root'
        edges = [(root_node, n) for n in roots]
        net.add_edges_from(edges)

        return net

    def info(self):
        return nx.info(self.netx)

    def get_tree(self):

        # convert to tree
        G = nx.bfs_tree(self.netx, 'root')

        return G

    def get_completed_network(self, n_levels):
        G = complete_network(self.netx, n_leveles=n_levels)
        return G

    def get_completed_tree(self, n_levels):
        G = self.get_tree()
        G = complete_network(G, n_leveles=n_levels)
        return G

    def get_layers(self, n_levels, direction='root_to_leaf'):
        if direction == 'root_to_leaf':
            net = self.get_completed_network(n_levels)
            layers = get_layers_from_net(net, n_levels)
        else:
            net = self.get_completed_network(5)
            layers = get_layers_from_net(net, 5)
            layers = layers[5 - n_levels:5]

        # get the last layer (genes level)
        terminal_nodes = [n for n, d in net.out_degree() if d == 0]  # set of terminal pathways
        # we need to find genes belonging to these pathways
        genes_df = self.reactome.pathway_genes

        dict = {}
        missing_pathways = []
        for p in terminal_nodes:
            pathway_name = re.sub('_copy.*', '', p)
            genes = genes_df[genes_df['group'] == pathway_name]['gene'].unique()
            if len(genes) == 0:
                missing_pathways.append(pathway_name)
            dict[pathway_name] = genes

        layers.append(dict)
        return layers


def insert_layer(layers, new_layer_df, input_col, output_col):
    """
    插入一个新的层到 layers 列表中，并确保层之间的关系一致。
    
    参数：
    layers (list of DataFrame): 包含各个层的数据框的列表。
    new_layer_df (DataFrame): 要插入的新的数据框。
    input_col (str): 新数据框的 input 列名称。
    output_col (str): 新数据框的 output 列名称。
    
    返回：
    list of DataFrame: 更新后的 layers 列表（不会修改原始 layers）。
    """
    
    # 1. 创建 layers 的深拷贝，避免修改原始 layers
    layers_copy = deepcopy(layers)
    
    # 2. 获取上一个层的 input_features 列（即最后一个层的 input_features）
    previous_input_features = layers_copy[-1]['input_features']
    
    # 3. 清洗新数据框：只保留 output_col 中在 previous_input_features 列中有交集的数据
    new_layer_df_filtered = new_layer_df[new_layer_df[output_col].isin(previous_input_features)]
    
    # 4. 重命名新数据框的列名，output_col 为 output_nodes，input_col 为 input_features
    new_layer_df_filtered = new_layer_df_filtered.rename(columns={output_col: 'output_nodes', input_col: 'input_features'})
    
    # 5. 将清洗后的新数据框插入 layers_copy 列表的最后
    layers_copy.append(new_layer_df_filtered)
    
    # 6. 从最后一个层开始往前遍历，确保 output_nodes 和 input_features 一致
    for i in range(len(layers_copy) - 1, 0, -1):
        # 当前层的 output_nodes 列
        current_output_nodes = layers_copy[i]['output_nodes']
        # 仅保留与当前 output_nodes 有交集的前一层的行
        layers_copy[i - 1] = layers_copy[i - 1][layers_copy[i - 1]['input_features'].isin(current_output_nodes)]
    
    return layers_copy


# 用法示例：
# 假设 layers 列表中已经有一些层，new_layer_df 是新的数据框
# layers = [layer1_df, layer2_df, layer3_df, layer4_df, layer5_df]
# new_layer_df = pd.DataFrame(...)  # 新的数据框
# 更新后的 layers
# layers = insert_layer(layers, new_layer_df, 'column_in_new_df_for_input', 'column_in_new_df_for_output')