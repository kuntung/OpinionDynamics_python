import leidenalg as la
import igraph as ig
import random
import numpy as np
import networkOpinionDynamics
import user_function


def show_opinion(opinion, g):
    clusters = la.find_partition(g, la.ModularityVertexPartition)
    member = clusters.membership
    new_cmap = ['#'+''.join([random.choice('0123456789abcdef') for x in range(6)])
                for z in range(len(clusters))]  # 得到每个社区的颜色类型
    # vcolors = {v: new_cmap[i] for i, c in enumerate(clusters) for v in c}  # key为节点社区编号，value为节点颜色
    vcolors = {v: [opinion[v-1], 1 - opinion[v-1], abs(0.5 - opinion[v-1])] for i, c in enumerate(clusters) for v in
                c}  # key为节点标签，value为节点颜色,根据观点值映射得到的节点颜色
    g.vs["color"] = [vcolors[v] for v in g.vs.indices]  # 遍历每个节点，得到他们的颜色值
    ecolors = {e.index: new_cmap[member[e.tuple[0]]] if member[e.tuple[0]] == member[e.tuple[1]] else "#e0e0e0" for e in
               g.es}  # 得到图的边颜色
    eweights = {e.index: (3 * g.vcount()) if member[e.tuple[0]] == member[e.tuple[1]] else 0.1 for e in g.es}
    g.es["weight"] = [eweights[e.index] for e in g.es]  # 将边集的权重赋值
    g.es["color"] = [ecolors[e] for e in g.es.indices]  # 将边集的颜色赋值
    ig.plot(g, make_groups=True).show()


if __name__ == '__main__':
    g = ig.Graph.Famous('Zachary')  # 生成Zachary图
    agent_num = g.vcount()  # 个体数目
    adj_matrix = np.array(g.get_adjacency().data)  # 将igraph的matrix转为numpy矩阵
    T = 20
    X0 = np.random.random((agent_num, ))
    opinion_result, action_result= networkOpinionDynamics.CODA_BC_network_opinion(X0, adj_matrix, 100, epsilonL=1, epsilonR=1)
    opinion = opinion_result[:, T-1]
    show_opinion(opinion, g)
    user_function.plot_opinion(opinion_result)




