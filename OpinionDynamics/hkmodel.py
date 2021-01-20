import numpy as np


np.set_printoptions(precision=4, suppress=True)

def BC_model(X0, T = 20, epsilonL = 0.25, epsilonR = 0.25):
    """经典的H-K model的Python实现

    Parameters
    ----------
    X0 : float
        The initial opinion profiles
    T : int
        Discrete time for communicating
    epsilonL: float
        The left bounded confidence level
    epsilonR : float
        The right bounded confidence level

    Notes
    -----
    通过给定不同的置信区间值以及交流周期，可以将交流结束后的观点值返回

    返回参数
    -------
    将交流结束后的观点，以`np.array()`的形式进行返回

    See Also
    --------
    find_neighbor()

    References
    ----------
    .. [1] Hegselmann, R., & Krause, U. (2002). Opinion Dynamics and Bounded Confidence Models,
    Analysis and Simulation. Journal of Artificial Societies and Social Simulation, 5(3), 1-2.

    """
    agents_num = X0.shape[0]
    X=np.zeros((agents_num, T))
    X[:, 0]=X0
    for i in range(1, T):#计算每一个时刻
        neighbor_set = find_neighbor(X[:, i-1].reshape(agents_num,), epsilonL, epsilonR)
        X[:, i] = np.dot(neighbor_set, X[:, i-1])/np.sum(neighbor_set, 1)
    return X

def find_neighbor(X_t, epsilonL = 0.3, epsilonR = 0.3):#根据观点值和信域区间确定观点邻近集
    """返回当前观点下的观点临近集

        Parameters
        ----------
        X_t : float
            当前观点值序列
        epsilonL: float
            The left bounded confidence level
        epsilonR : float
            The right bounded confidence level

    """

    agents_num = X_t.shape[0]
    neighbor_set = np.zeros((agents_num, agents_num))
    for i in range(agents_num):
        difference_x = X_t - X_t[i]
        neighbor_set[i, :] = ((difference_x >= -epsilonL) & (difference_x <= epsilonR))

    return neighbor_set


if __name__=='__main__':
    #调用代码
    T = 20
    agents_num = 200
    np.random.seed(10)
    X0 = np.random.rand(agents_num,)
    result_x = BC_model(X0, T)
    print(result_x.round(3))
    plot_opinion(result_x)