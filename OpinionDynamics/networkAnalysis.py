import networkx as nx
import pandas as pd

def network_info(G):
    """用于计算生成网络的基本结构属性

    返回参数
    -------
    字典类型数据：{'diameter':*,'density':*,'number_of_nodes':*,'number_of_edges':*}

    """
    info = dict(average_clustering=nx.average_clustering(G))
    info['diameter'] = nx.diameter(G)
    info['density'] = nx.density(G)
    info['number_of_nodes'] = nx.number_of_nodes(G)
    info['number_of_edges'] = nx.number_of_edges(G)

    return info

if __name__ == '__main__':
    T = 20
    agents_num = 200
    k = 20
    p = 0.7
    seed = 20532
    G = nx.generators.random_graphs.newman_watts_strogatz_graph(agents_num, k, p, seed=seed)
    G1 = nx.generators.random_graphs.newman_watts_strogatz_graph(agents_num, 50, 0.3, seed=seed)
    info_G = network_info(G)
    info_G1 = network_info(G1)
    df = pd.DataFrame([info_G, info_G1])