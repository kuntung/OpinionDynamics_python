import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkOpinionDynamics
import scipy.io as sio
import user_function

def analyze_opinion(opinion_profile):
    """计算和分析群体观点的相关数据

    Parameters
    ----------
    opinion_profile : array_like
    群体观点分布集
    """
    (agents_num, T) = opinion_profile.shape
    df_index = ['agent{}'.format(i) for i in range(1, agents_num+1)]
    df_columns = ['T={}'.format(i) for i in range(1, T+1)]
    opinion_df = pd.DataFrame(opinion_profile)
    opinion_df.index = df_index
    opinion_df.columns = df_columns
    stable_time, total_area = cal_area(opinion_profile)
    group_cost1, group_cost2 = cal_index(opinion_profile)
    opinion_df['Area'] = total_area
    opinion_df['Cost1'] = group_cost1
    opinion_df['Cost2'] = group_cost2
    opinion_df['StableTime'] = stable_time

    return opinion_df


def stable_type(list_opinion):
    """根据观点值演化趋势，选择面积的计算标准
    Parameters
        ----------
        list_opinion : list_like
        
    Returns
        -------
        array_like
    """
    opinion_arr = np.zeros(len(list_opinion),)
    if (list_opinion[-1]) > 0.5:
        opinion_arr = 1 - np.array(list_opinion)
    else:
        if (list_opinion[-1]) < 0.5:
            opinion_arr = np.array(list_opinion)
        else:
            if list_opinion[0] > 0.5 and (list_opinion[-1]) == 0.5:
                opinion_arr = np.array(list_opinion) - 0.5
            else:
                if list_opinion[0] < 0.5 and (list_opinion[-1]) == 0.5:
                    opinion_arr = 0.5 - np.array(list_opinion)

    return opinion_arr


def cal_area(opinion_profile):
    """用来计算每个个体到达观点值稳定时的速度、包围面积、时间

    Parameters
    ----------
    opinion_profile : array_like
    交流结束后的群体观点集合

    Notes
    -----
    对于传入的观点演化数据，我们认为当个体的观点值前后不再发生改变的时候，该个体达到自我的观点稳态。可能属于polarization, fragmentation或
    consensus。但是无论如何，从交流初期到达这一状态的过程中，存在着一个时间周期stable_time。并且，与稳态观点值横线总能围成一个不规则图形，
    通过分析可以发现，如果一个个体越早的达到稳态值，那么他所包围的面积就越可能的小。所以，我们可以通过判断达到群体达到稳态时的面积之和来判断这个
    群体的观点快速性
    Returns
    -------

    """
    (agents_num, T) = opinion_profile.shape
    stable_time = np.zeros(agents_num, ).astype(int)
    stable_time = list(map(lambda x: x+T, stable_time))  # 将观点值达到稳定的时间初始化为T
    total_area = np.zeros(agents_num,)  # 初始化所有个体的覆盖面积
    for i in range(agents_num):
        list_opinion = list(opinion_profile[i, :])  # 得到当前个体的所有时刻观点值列表
        stable_time[i] = list_opinion.index(list_opinion[-1])  # 计算每个个体观点值的实际稳定时间
        opinion_arr = abs(stable_type(list_opinion))
        total_area[i] = sum(opinion_arr) - opinion_arr[0]/2 - opinion_arr[stable_time[i]]/2

    return stable_time, total_area.round(3)


def cal_index(opinion_profile):
    """用来计算群体的观点演化指标：快速性、一致性等

    Parameters
    ----------
    opinion_profile : array_like
    群体的观点集合

    Notes
    -----
    在`cal_area()`中，我们交流结束后的观点集合做了一些基本的统计。比如：
    1. 计算每个个体在达到稳定过程中所覆盖的面积
    2. 达到稳定过程所用的时间
    3. 在整个过程中的平均面积，即cost
    通过这些数据，我们可以通过cost很直观地看出每个个体的观点稳定速度。但是对于不同的群体，我们希望能够有个更加通用的、能够考虑全部个体影响的指标
    来进行定量分析。因此，我们需要计算他们的群体cost
    """
    (agents_num, T) = opinion_profile.shape
    stable_time, total_area = cal_area(opinion_profile)
    # 快速性
    group_cost1 = np.sum(total_area)/np.sum(stable_time)  # 总的面积除以总的稳定时间
    cost = np.zeros(agents_num,)
    for i in range(agents_num):
        if stable_time[i] == 0:
            cost[i] = total_area[i]/T
        else:
            cost[i] = total_area[i]/stable_time[i]
    group_cost2 = np.sum(cost)

    return group_cost1, group_cost2


if __name__ == '__main__':
    T = 20
    agents_num = 200
    k = 20
    p = 0.7
    seed = 20532
    alpha = 0.7
    beta = 0.7
    act_threshold = 0.5
    data = sio.loadmat('./testData.mat')
    adj_matrix = np.array(data['NB'].todense())
    X0 = data['X0'].reshape(200, )
    x_result1, act_result1 = networkOpinionDynamics.CODA_BC_network_opinion(X0, adj_matrix, T, epsilonL=0.1,
                                                                          epsilonR=0.25)

    group_cost11, group_cost12 = cal_index(x_result1)
    x_result2, act_result2 = networkOpinionDynamics.CODA_BC_network_opinion(X0, adj_matrix, T, epsilonL=0.1,
                                                                          epsilonR=0.4)
    group_cost21, group_cost22 = cal_index(x_result2)

    x_result3, act_result3 = networkOpinionDynamics.CODA_BC_network_opinion(X0, adj_matrix, T, epsilonL=0.1,
                                                                          epsilonR=0.45)
    group_cost31, group_cost32 = cal_index(x_result3)

    # 绘制图像
    user_function.plot_opinion(x_result1)
    user_function.plot_opinion(x_result2)
    user_function.plot_opinion(x_result3)

    # 得到数据流
    opinion_df1 = analyze_opinion(x_result1)
    opinion_df2 = analyze_opinion(x_result2)
    opinion_df3 = analyze_opinion(x_result3)

