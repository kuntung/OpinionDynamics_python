import numpy as np
import networkx as nx
import user_function
from hkmodel import find_neighbor
import scipy.io as sio
import OpinionAnalysis

def network_opinion(adj_matrix, X0, T):
    """进行社会网络下的观点演化分析

        Parameters
        ----------
        adj_matrix : np.array
            当前网络结构的邻接矩阵
        X0 : float
            初始观点集
        T : int
            交流周期

        Notes
        -----
        对一个社会网络，将其邻接矩阵传入，其中邻接矩阵中非0元素代表两个个体能够进行观点交流。
        进而，随着时间T的变化，个体的观点值也会得到演化更新。
    """
    assert adj_matrix.shape[0] == X0.shape[0], 'the shape of adj_matrix should equal to X0'
    agents_num = X0.shape[0]
    X = np.zeros((agents_num, T))
    X[:, 0] = X0
    for i in range(1, T):  # 计算每一个时刻
        X[:, i] = np.dot(adj_matrix, X[:, i - 1]) / np.sum(adj_matrix, 1)
    return X.round(3)


def BC_netwrk_opinion(adj_matrix, X0, T, epsilonL=0.3, epsilonR=0.3):
    """进行社会网络下的有限信任观点演化分析

            Parameters
            ----------
            adj_matrix : np.array
                当前网络结构的邻接矩阵
            X0 : float
                初始观点集
            T : int
                交流周期
            epsilonL: float
                The left bounded confidence level
            epsilonR : float
                The right bounded confidence level

            Notes
            -----
            对一个社会网络，将其邻接矩阵传入，其中邻接矩阵中非0元素代表两个个体能够进行有机会观点交流。
            而根据有限信任模型，两个个体会选择交流观点当且仅当他们的观点在一定程度上接近、相似
            所以，本函数用来分析在社会网络中，存在着信任区间的情况下的观点演化

            """
    assert adj_matrix.shape[0] == X0.shape[0], 'the shape of adj_matrix should equal to X0'
    agents_num = X0.shape[0]
    X = np.zeros((agents_num, T))
    X[:, 0] = X0
    for i in range(1, T):  # 计算每一个时刻
        neighbor_set = find_network_neighbor(adj_matrix, X[:, i - 1], epsilonL, epsilonR)
        X[:, i] = np.dot(neighbor_set, X[:, i - 1]) / np.sum(adj_matrix, 1)

    return X.round(3)


def CODA_rule(xi, action, alpha, beta):
    """用来实现每个个体的观点更新

    Parameters
    ----------
    xi : float
        个体i在当前时刻的观点值
    action : int
        参考个体的当前时刻行为意愿
    alpha : float
    beta : float

    Returns
    -------
    返回个体i在参考其他个体后的观点更新值
    """
    Flag = -1
    if action == 1:
        Flag = 1
    else:
        if action == -1:
            Flag = 2

    if xi == 1:
        Flag = 3  # 观点值已经最明显为1
    else:
        if xi == 0:
            Flag = 4

    if Flag == 1:
        O1 = xi/(1-xi)*alpha/(1-beta)
        xi_new = O1/(1+O1)
    else:
        if Flag == 2:
            O2 = xi/(1-xi)*(1-alpha)/beta
            xi_new = O2/(1+O2)
        else:
            if Flag == 3:
                xi_new = 1
            else:
                xi_new = 0

    return xi_new


def network_CODA_rule(xi, action, alpha, beta):
    """用来实现BC&CODA中每个个体的观点更新

    Parameters
    ----------
    xi : float
        个体i在当前时刻的观点值
    action : int
        参考个体的当前时刻行为意愿
    alpha : float
    beta : float

    Returns
    -------
    返回个体i在参考其他个体后的观点更新值
    """
    Flag = -1
    if action == 1:
        Flag = 1
    else:
        if action == -1:
            Flag = 2

    if xi == 1 and Flag == 1:
        Flag = 3  # 观点值无法增加情形
    else:
        if xi == 0 and Flag == 2:
            Flag = 4  # 观点值无法减小情形

    if Flag == 1:
        O1 = xi/(1-xi)*alpha/(1-beta)
        xi_new = O1/(1+O1)
    else:
        if Flag == 2:
            O2 = xi/(1-xi)*(1-alpha)/beta
            xi_new = O2/(1+O2)
        else:
            if Flag == 3:
                xi_new = 1
            else:
                xi_new = 0

    return xi_new


def CODA_model(X0, T, alpha=0.7, beta=0.7, act_threshold=0.5):
    """原始的CODA准则实现

        Parameters
        ----------
        X0 : np.array
            初始观点集
        T : int
            交流周期
        alpha : float
            在观点偏向于A的时候，认为观察个体也如此的概率
        beta : float
            在观点偏向于B的时候，认为观察个体也如此的概率
        act_threshold : float
            观点触发阈值

        Notes
        -----
        在该模型中，每个个体在每一个时刻会将一个随机个体的行为作为观点更新的参考

        Returns
        -------
        交流后的观点值分布和行为选择情况

        References
        ----------
        Martins, A. C. (2008). Continuous opinions and discrete actions in
        opinion dynamics problems. International Journal of Modern Physics C, 19(04), 617-624.
    """
    agents_num = X0.shape[0]
    X = np.zeros((agents_num, T))
    X[:, 0] = X0
    Action = np.zeros((agents_num, T))
    Action[X0 >= act_threshold, 0] = 1
    Action[X0 < act_threshold, 0] = -1
    for i in range(1, T):
        for j in range(agents_num):
            actor = user_function.randint_digit(0, agents_num, j)
            X[j, i] = CODA_rule(X[j, i-1], Action[actor, i-1], alpha, beta)
        Action[X[:, i] >= act_threshold, i] = 1
        Action[X[:, i] < act_threshold, i] = -1

    return X.round(3), Action

def CODA_network_opinion(X0, adj_matrix, T, alpha=0.7, beta=0.7, act_threshold=0.5):
    """个体会和当前网络中所有连接个体进行观点交流

    Parameters
    ----------
    X0: np.array
        初始观点集
    adj_matrix : np.array
        社会网络的邻接矩阵
    T : int
        交流周期
    alpha : float
        在观点偏向于A的时候，认为观察个体也如此的概率
    beta : float
        在观点偏向于B的时候，认为观察个体也如此的概率
    act_threshold : float
        观点触发阈值

    Notes
    -----
    在固定的社会网络结构中，一个个体会受到与他存在物理连接的其他个体的行为影响。
    而不是像之前的CODA准则中，随机选择一个个体作为参考

    Returns
    -------
    返回交流结束后的观点态度以及行为意愿
    """
    agents_num = X0.shape[0]
    X = np.zeros((agents_num, T))
    X[:, 0] = X0
    Action = np.zeros((agents_num, T))
    Action[X0 >= act_threshold, 0] = 1
    Action[X0 < act_threshold, 0] = -1
    for i in range(1, T):
        for j in range(agents_num):
            P1A1 = CODA_rule(X[j, i - 1], 1, alpha, beta)
            P1A0 = CODA_rule(X[j, i - 1], -1, alpha, beta)
            neighbor_set = adj_matrix[j, :]  # 当前个体j的邻近集
            action = Action[:, i-1]  # 当前时刻，所有个体的行为意愿
            neighbor_action = action * neighbor_set
            neighbor_action[neighbor_action == -1, ] = P1A0
            neighbor_action[neighbor_action == 1, ] = P1A1
            X[j, i] = np.dot(neighbor_set, neighbor_action)/np.sum(neighbor_set)
        Action[X[:, i] >= act_threshold, i] = 1
        Action[X[:, i] < act_threshold, i] = -1

    return X.round(3), Action


def CODA_BC_network_opinion(X0, adj_matrix, T, epsilonL = 0.3, epsilonR = 0.3,
                            alpha=0.7, beta=0.7, act_threshold=0.5):
    """返回社会网络下存在着信任交互的CODA观点行为演化结果

    Parameters
    ----------
    X0 : np.array
        初始观点集
    adj_matrix : np.array
        邻接矩阵
    T : int
        交流周期
    epsilonL : float
        信任水平
    epsilonR : float
        信任水平
    alpha : float
    在观点偏向于A的时候，认为观察个体也如此的概率
    beta : float
        在观点偏向于B的时候，认为观察个体也如此的概率
    act_threshold : float
        观点触发阈值

    Notes
    -----
    对于当前时刻的个体i，ta对于某个话题的观点、态度会受到社会网络下存在着关联的其他个体的行为影响。
    但是结合bounded confidence model，并非所有相连接的个体行为都会对个体i的观点产生影响。
    而是，那些不仅有着关联，而且观点值相近，即满足bounded confidence model的个体才能够对个体i的
    观点值产生影响
    """
    agents_num = X0.shape[0]
    X = np.zeros((agents_num, T))
    X[:, 0] = X0
    Action = np.zeros((agents_num, T))
    Action[X0 >= act_threshold, 0] = 1
    Action[X0 < act_threshold, 0] = -1
    for i in range(1, T):
        neighbor_set = find_network_neighbor(adj_matrix, X[:, i - 1], epsilonL, epsilonR)
        for j in range(agents_num):
            P1A1 = network_CODA_rule(X[j, i - 1], 1, alpha, beta)
            P1A0 = network_CODA_rule(X[j, i - 1], -1, alpha, beta)
            neighbor = neighbor_set[j, :]  # 当前个体j的邻近集
            action = Action[:, i - 1]  # 当前时刻，所有个体的行为意愿
            neighbor_action = action * neighbor
            neighbor_action[neighbor_action == -1, ] = P1A0
            neighbor_action[neighbor_action == 1, ] = P1A1
            if np.count_nonzero(neighbor_action) == 0:
                X[j, i] = X[j, i-1]
            else:
                X[j, i] = np.sum(neighbor_action)/np.count_nonzero(neighbor_action)
        Action[X[:, i] >= act_threshold, i] = 1
        Action[X[:, i] < act_threshold, i] = -1

    return X.round(3), Action


def find_network_neighbor(adj_matrix, X_t, epsilonL=0.3, epsilonR=0.3):
    """返回一个和邻接矩阵规模相同的矩阵，其中非0元素表示存在着物理链接，且观点相近

           Parameters
           ----------
           adj_matrix : np.array
               当前网络结构的邻接矩阵
           X0 : float
               初始观点集
           epsilonL: float
               The left bounded confidence level
           epsilonR : float
               The right bounded confidence level

           Notes
           -----
           根据当前的观点值，判定网络中和自身观点相近个体集合

           """
    opinion_neighbor = find_neighbor(X_t, epsilonL, epsilonR)
    neighbor_set = opinion_neighbor * adj_matrix

    return neighbor_set


if __name__ == '__main__':
    # T = 100
    # agents_num = 200
    # k = 20
    # p = 0.7
    # seed = 20532
    # alpha = 0.7
    # beta = 0.7
    # act_threshold = 0.5
    #生成自定义邻接矩阵
    T = 20
    agents_num = 200
    k = 20
    p = 0.7
    seed = 20532
    alpha = 0.7
    beta = 0.7
    act_threshold = 0.5
    # 生成自定义邻接矩阵
    # G = nx.generators.random_graphs.newman_watts_strogatz_graph(agents_num, k, p)
    # G = nx.generators.random_graphs.barabasi_albert_graph(agents_num, 3)
    # adj_matrix = np.array(nx.adj_matrix(G).todense())
    # # np.random.seed(seed)
    # X0 = np.random.rand(agents_num, )
    # # result_x = BC_netwrk_opinion(adj_matrix, X0, T)
    # # print(result_x.round(3))
    # # plot_opinion(result_x)
    # x_result, act_result = CODA_network_opinion(X0, adj_matrix, T, alpha=0.7, beta=0.7, act_threshold=0.3)
    # user_function.plot_opinion(x_result)

    # 应用论文仿真相同的邻接矩阵。生成算法为barabasi_albert_graph
    # data = sio.loadmat('./testData.mat')
    # adj_matrix = np.array(data['NB'].todense())
    # X0 = data['X0'].reshape(200, )
    # x_result, act_result = CODA_network_opinion(X0, adj_matrix, T, alpha=0.7, beta=0.7, act_threshold=0.5)
    # user_function.plot_opinion(x_result)

    # 真实的social network
    data = sio.loadmat('./testData.mat')
    adj_matrix = np.array(data['NB'].todense())
    X0 = data['X0'].reshape(200, )
    x_result, act_result = CODA_BC_network_opinion(X0, adj_matrix, T, epsilonL=0.1, epsilonR=0.1)
    user_function.plot_opinion(x_result)
    user_function.plot_opinion_diff(x_result)

    # 对观点结果进行分析
    result_df = OpinionAnalysis.analyze_opinion(x_result)
