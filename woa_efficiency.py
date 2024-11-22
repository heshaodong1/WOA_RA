import math
import random

import numpy as np
from matplotlib import pyplot as plt

from xiangnong import cal_rate_feasible
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定默认字体：解决plot不能显示中文问题
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
distances = [[150, 150, 250, 180, 120, 155, 167, 300, 200, 140],
             [230, 130, 280, 50, 110, 190, 268, 90, 70, 200]]  # 用户与每个基站的距离


class woa():
    def __init__(self, X_train, LB, UB, b=1, whale_num=20, max_iter=500):
        self.LB = LB
        self.UB = UB
        self.whale_num = whale_num
        self.max_iter = max_iter
        self.b = b
        # 初始化鲸鱼位置
        self.X = X_train
        self.gBest_score = 0
        self.gBest_curve = np.zeros(max_iter)
        self.gBest_X = self.X[0]

    # 优化模块
    def opt(self):
        t = 0
        best_fitness_value = []
        efficiency_record = []
        efficiency_record_temp = 0
        # 初始化每条鲸鱼的位置
        for i in range(self.whale_num):
            for j in range(len(self.X[i])):
                for k in range(len(self.X[i][j])):
                    if self.X[i][j][k] != 0:
                        self.X[i][j][k] += np.random.uniform(-2, 2)
                        if self.X[i][j][k] < self.LB:
                            self.X[i][j][k] = self.LB
                        if self.X[i][j][k] > self.UB:
                            self.X[i][j][k] = self.UB

        while t < self.max_iter:
            a = 2 * (self.max_iter - t) / self.max_iter
            # 更新鲸鱼位置
            for i in range(self.whale_num):
                # 分条件进行更新
                p = np.random.uniform()
                R1 = np.random.uniform(size=(station_number, user_number))
                R2 = np.random.uniform(size=(station_number, user_number))
                A = 2 * a * R1 - a
                C = 2 * R2
                l = 2 * np.random.uniform(size=(station_number, user_number)) - 1

                if p >= 0.5:
                    D = abs(self.gBest_X - self.X[i, :])
                    self.X[i, :] = D * np.exp(self.b * l) * np.cos(2 * np.pi * l) + self.gBest_X
                else:
                    if np.sum(np.sum(np.abs(A))) < station_number * user_number:
                        D = abs(C * self.gBest_X - self.X[i, :])
                        self.X[i, :] = self.gBest_X - A * D
                    else:
                        rand_index = np.random.randint(low=0, high=self.whale_num)
                        X_rand = self.X[rand_index, :]
                        D = abs(C * X_rand - self.X[i, :])
                        self.X[i, :] = X_rand - A * D

                for row_index in range(len(self.X[i, :])):
                    for col_index in range(len(self.X[i, row_index])):
                        if self.X[i][row_index][col_index] < self.LB and self.X[i][row_index][col_index] != 0:
                            self.X[i][row_index][col_index] = self.LB
                        if self.X[i][row_index][col_index] > self.UB:
                            self.X[i][row_index][col_index] = self.UB

            for i in range(self.whale_num):
                sum_rate, gain = fit_function(self.X[i, :])
                if sum_rate > self.gBest_score:
                    self.gBest_score = sum_rate
                    self.gBest_X = self.X[i, :].copy()
                    efficiency_record_temp = gain
            efficiency_record.append(efficiency_record_temp)
            best_fitness_value.append(self.gBest_score)
            self.gBest_curve[t] = self.gBest_score
            t += 1
        return self.gBest_curve, best_fitness_value, efficiency_record


def fit_function(channel):
    total_user_rate, violation_rate = cal_rate_feasible(channel, distances)
    return np.sum(total_user_rate), np.sum(violation_rate)


if __name__ == '__main__':
    station_number = 2
    user_number = 10
    whale_num = 5
    iterators = 200  # 迭代次数
    channel_user = np.zeros([whale_num, station_number, user_number])
    for whale in range(whale_num):
        channel_user[whale] = [[29, 30, 32, 28, 25, 34, 28, 25, 27, 30],
                               [28, 26, 30, 34, 34, 35, 26, 27, 30, 33]]

    fitnessCurve, best_fitness, efficiency = woa(channel_user, LB=1, UB=35,
                                     whale_num=whale_num, max_iter=iterators).opt()

    fig, ax1 = plt.subplots(figsize=(12, 10), dpi=300)
    plt.title('网络传输速率和不满足传输速率', fontdict={'weight': 'normal', 'size': 30})
    x = range(0, iterators, 1)
    ax1.plot(x, efficiency, color="red", label="不满足的传输速率", linewidth=3.0, linestyle="-")
    ax2 = ax1.twinx()
    ax2.plot(x, best_fitness, color="blue", label="网络传输速率", linewidth=3.0, linestyle="-")
    plt.tick_params(labelsize=25)
    plt.xlabel("迭代次数", fontdict={'weight': 'normal', 'size': 30})
    ax1.set_ylabel("不满足速率", fontdict={'weight': 'normal', 'size': 30})
    ax2.set_ylabel("传输速率", fontdict={'weight': 'normal', 'size': 30})
    plt.xticks(range(0, iterators, 20))

    # 设置图例
    line1, =plt.plot([1], label="不满足的传输速率", color="red")
    line2, =plt.plot([1], label="网络传输速率", color="blue")
    plt.rcParams.update({'font.size': 25})
    plt.legend(handles=[line1, line2], loc='upper right')
    plt.show()
