# @Time    :2020/12/12 11:40
# @Author  :korolTung
# @FileName: user_function.py
import numpy as np
import matplotlib.pyplot as plt

def randomint_plus(low, high=None, discard=None, size=None):
    """用来生成不包含一些中间数值的随机整数或序列

    Parameters
    ----------
    low : int
        生成随机整数的下界
    high: int
        生成随机整数的上界
    discard: int/list
        不需要包含的一个或多个值
    size: tuple
        需要产生的随机数规模
    Notes
    -----
    1. 在调用过程中，如果high, discard, size都缺省，就默认返回一个[0, low)的值
    2. 如果discard, size缺省，返回[low, high)的一个随机整数值
    3. 如果size缺省， 则返回一个[low, discard)U(discard, high)的随机整数值
    4. 返回一个给定size的矩阵，矩阵元素为[low, discard)U(discard, high)的随机整数值

    See Also
    --------
    np.random.randint()

    """
    if high is None:
        assert low is not None, "必须给定一个值作为上界"
        high = low
        low = 0
    number = 1  # 将size拉长成为一个向量
    if size is not None:
        for i in range(len(size)):
            number = number * size[i]

    if discard is None:  # 如果不需要剔除值，就通过numpy提供的函数生成随机整数
        random_result = np.random.randint(low, high, size=size)
    else:
        if number == 1:  # 返回一个随机整数
            random_result = randint_digit(low, high, discard)
        else:  # 返回一个形状为size的随机整数数组
            random_result = np.zeros(number, )
            for i in range(number):
                random_result[i] = randint_digit(low, high, discard)
            random_result = random_result.reshape(size)

    return random_result.astype(int)


def randint_digit(low, high, discard):
    """用来生成一个在区间[low, high)排除discard后的随机整数

    Parameters
    ----------
    low: int
        下限，能够取到
    high: int
        上限，不能够取到
    discard: int/list
        一个需要剔除的数或者数组，要求在(low, high)区间之间
    """
    digit_list = list(range(low, high))
    if type(discard) is int:  # 只需要剔除一个值
        if discard in digit_list:  # 如果需要剔除的值不存在，则不执行剔除操作
            digit_list.remove(discard)
    else:
        for i in discard:  # 需要剔除多个值的情况
            if i not in digit_list:  # 如果需要剔除的值不存在，则不执行剔除操作
                continue
            digit_list.remove(i)

    np.random.shuffle(digit_list)

    return digit_list.pop()  # 生成的序列打乱并且返回当前的随机值

def plot_opinion(opinion_profile):#定义绘制观点动力学函数
    """绘制每个个体的观点曲线图

      Parameters
      ----------
      opinion_profile : array_like
          观点集合

    """
    (agents_num, T) = opinion_profile.shape
    plt.figure(figsize=(20, 8), dpi=400)
    x = range(0, T)
    X0=opinion_profile[:, 0]
    for i in range(agents_num):
        plt.plot(range(0, T), opinion_profile[i, :], color=[X0[i], abs(abs(1-X0[i])), abs(0.5-X0[i])])
    plt.xlabel('time')
    plt.ylabel('Opinion value')
    x_ticklabels = range(1, T+1)
    plt.xticks(x, x_ticklabels)
    plt.show()


def plot_opinion_diff(opinion_profile):#定义绘制观点动力学函数
    """绘制观点变化曲线图

    Parameters
    ----------
    opinion_profile : array_like
        观点集合

    Notes
    -----
    通过传入当前观点集合，计算得到每个个体前后两时刻之间的观点变化值

    See Also
    --------
    np.diff()
    """
    opinion_diff_profile = np.diff(opinion_profile)
    (agents_num, T) = opinion_diff_profile.shape
    plt.figure(figsize=(20, 8), dpi=400)
    x = range(0, T)
    X0=opinion_profile[:, 0]
    for i in range(agents_num):
        plt.plot(range(0, T), opinion_diff_profile[i, :], color=[abs(X0[i]), abs(1-abs(X0[i])), abs(0.5-X0[i])])
    plt.xlabel('time')
    plt.ylabel('Opinion value')
    x_ticklabels = range(1, T+1)
    plt.xticks(x, x_ticklabels)
    plt.show()


if __name__ == '__main__':
    discard1 = 3
    discard2 = [3, 5, 10]
    low = 1
    high = 9
    result = randint_digit(low, high, discard2)