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
        archive_size = 10
        best_fitness_value = []
        # 档案集个体是否被支配 1未被支配
        archive_flag = []
        # 档案集存放发射功率
        archive_set = []
        # 档案集对应的传输速率
        archive_rate = []
        # 档案集对应的能量效率
        archive_efficiency = []
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
                        while self.X[i][row_index][col_index] < self.LB or self.X[i][row_index][col_index] > self.UB:
                            while self.X[i][row_index][col_index] < self.LB:
                                self.X[i][row_index][col_index] = self.X[i][row_index][col_index] + 2 * math.exp(-(self.max_iter - t)/self.max_iter) * (self.LB - self.X[i][row_index][col_index])
                            while self.X[i][row_index][col_index] > self.UB:
                                self.X[i][row_index][col_index] = self.X[i][row_index][col_index] - 2 * math.exp(-(self.max_iter - t)/self.max_iter) * (self.X[i][row_index][col_index] - self.UB)
                        # if self.X[i][row_index][col_index] < self.LB and self.X[i][row_index][col_index] != 0:
                        #     self.X[i][row_index][col_index] = self.LB
                        # if self.X[i][row_index][col_index] > self.UB:
                        #     self.X[i][row_index][col_index] = self.UB

            all_rate = []
            all_gain = []
            for i in range(self.whale_num):
                sum_rate, gain = fit_function(self.X[i, :])
                all_rate.append(sum_rate)
                all_gain.append(gain)

            for i in range(self.whale_num):
                # 如果档案集大小不够
                dominate_position = 0
                dominate_flag = False
                # a支配b的数量
                dominate_count = 0
                # a被b支配的数量
                dominated_count = 0
                # 如果当前档案集没有个体
                if len(archive_set) == 0:
                    for whale_index in range(self.whale_num):
                        archive_set.append(self.X[whale_index, :].copy())
                        archive_rate.append(all_rate[whale_index].copy())
                        archive_efficiency.append(all_gain[whale_index])
                        archive_flag.append(1)
                for archive_index in range(len(archive_set)):
                    for station_index in range(station_number):
                        if (np.sum(all_gain[i][station_index])) < np.sum(archive_efficiency[archive_index][station_index]):
                            dominate_count += 1
                        else:
                            dominated_count += 1
                    # 如果a支配的数量大于b
                    if np.sum(all_rate[i]) > np.sum(archive_rate[archive_index]):
                        dominate_count += 1
                        if dominate_count >= dominated_count:
                            print('有支配解')
                            archive_flag[archive_index] = 0
                            dominate_flag = True
                            dominate_position = archive_index
                # 如果a支配了档案集中的个体
                if dominate_flag:
                    archive_flag.insert(dominate_position + 1, 1)
                    archive_set.insert(dominate_position + 1, self.X[i, :].copy())
                    archive_rate.insert(dominate_position + 1, all_rate[i].copy())
                    archive_efficiency.insert(dominate_position + 1, all_gain[i])
                # 如果当前档案集数量已满
                if len(archive_set) > archive_size:
                    archive_rate_sum = []
                    archive_efficiency_sum = []
                    for archive_index in range(len(archive_set)):
                        archive_rate_sum.append(np.sum(archive_rate[archive_index]))
                        archive_efficiency_sum.append(np.sum(archive_efficiency[archive_index]))
                    max_rate = np.max(archive_rate_sum) - np.min(archive_rate_sum)
                    max_efficiency = np.max(archive_efficiency_sum) - np.min(archive_efficiency_sum)
                    # 挨个计算距离，删除最远距离的个体
                    distances = []
                    max_distance = 0
                    max_distance_index = 0
                    for archive_index in range(len(archive_set) - 1):
                        distance_temp = math.sqrt((math.pow(abs(np.sum(archive_rate[archive_index]) - np.sum(
                            archive_rate[archive_index + 1])) / max_rate, 3))
                                                  + (abs(math.pow((np.sum(archive_efficiency[archive_index]) - np.sum(
                            archive_efficiency[archive_index + 1])) / max_efficiency, 2))))
                        distances.append(distance_temp)
                        if distance_temp > max_distance and archive_flag[archive_index] == 0:
                            max_distance = distance_temp
                            max_distance_index = archive_index
                    archive_set.pop(max_distance_index)
                    archive_rate.pop(max_distance_index)
                    archive_efficiency.pop(max_distance_index)
                    archive_flag.pop(max_distance_index)

                # 计算种群偏离方向
                efficiency_sum = []
                rate_sum = []
                for archive_index in range(len(archive_set)):
                    efficiency_sum.append(np.sum(archive_efficiency[archive_index]))
                    rate_sum.append(np.sum(archive_rate[archive_index]))
                if np.sum(efficiency_sum) == 0:
                    slope, intercept = 0, 0
                else:
                    slope, intercept = np.polyfit(efficiency_sum, rate_sum, 1)
                lower_count = 0
                lower_efficiency = []
                lower_rate = []
                lower_flag = []
                upper_count = 0
                upper_efficiency = []
                upper_rate = []
                upper_flag = []
                for set_index in range(len(archive_set)):
                    if np.sum(archive_efficiency[set_index]) * slope + intercept < np.sum(archive_rate[set_index]):
                        lower_count += 1
                        lower_efficiency.append(archive_efficiency[set_index])
                        lower_rate.append(archive_rate[set_index])
                        lower_flag.append(archive_flag[set_index])
                    else:
                        upper_count += 1
                        upper_efficiency.append(archive_efficiency[set_index])
                        upper_rate.append(archive_rate[set_index])
                        upper_flag.append(archive_flag[set_index])

                lower_rand_position = []
                for flag_index in range(len(lower_flag)):
                    if lower_flag[flag_index] != 0:
                        lower_rand_position.append(flag_index)
                upper_rand_position = []
                for flag_index in range(len(upper_flag)):
                    if upper_flag[flag_index] != 0:
                        upper_rand_position.append(flag_index)
                # 根据偏离方向选择重视值
                if lower_count > upper_count:
                    # 重视upper
                    if len(lower_rand_position) == 0:
                        self.gBest_X = (1.5 * archive_set[upper_rand_position[random.randint(0, len(upper_rand_position) - 1)]] + 0.5 * archive_set[random.randint(0, len(archive_set) - 1)]) / 2
                    elif len(upper_rand_position) == 0:
                        self.gBest_X = (1.5 * archive_set[random.randint(0, len(archive_set) - 1)] + 0.5 * archive_set[lower_rand_position[random.randint(0, len(lower_rand_position) - 1)]]) / 2
                    else:
                        self.gBest_X = (1.5 * archive_set[upper_rand_position[random.randint(0, len(upper_rand_position) - 1)]] + 0.5 * archive_set[lower_rand_position[random.randint(0, len(lower_rand_position) - 1)]]) / 2
                    self.gBest_score, efficiency_record_temp = fit_function(self.gBest_X)
                else:
                    if len(lower_rand_position) == 0:
                        self.gBest_X = (0.5 * archive_set[upper_rand_position[random.randint(0, len(upper_rand_position) - 1)]] + 1.5 * archive_set[random.randint(0, len(archive_set) - 1)]) / 2
                    elif len(upper_rand_position) == 0:
                        self.gBest_X = (0.5 * archive_set[random.randint(0, len(archive_set) - 1)] + 1.5 * archive_set[lower_rand_position[random.randint(0, len(lower_rand_position) - 1)]]) / 2
                    else:
                        self.gBest_X = (0.5 * archive_set[upper_rand_position[random.randint(0, len(upper_rand_position) - 1)]] + 1.5 * archive_set[lower_rand_position[random.randint(0, len(lower_rand_position) - 1)]]) / 2
                    self.gBest_score, efficiency_record_temp = fit_function(self.gBest_X)
            efficiency_record.append(np.sum(efficiency_record_temp))
            best_fitness_value.append(np.sum(self.gBest_score))
            self.gBest_curve[t] = np.sum(self.gBest_score)
            t += 1
        return self.gBest_curve, best_fitness_value, efficiency_record


def fit_function(channel):
    total_user_rate, violation_rate = cal_rate_feasible(channel, distances)
    return total_user_rate, violation_rate


if __name__ == '__main__':
    station_number = 2
    user_number = 10
    whale_num = 10
    iterators = 100  # 迭代次数
    channel_user = np.zeros([whale_num, station_number, user_number])
    for whale in range(whale_num):
        channel_user[whale] = [[29, 30, 32, 28, 25, 34, 28, 25, 27, 30],
                               [28, 26, 30, 34, 34, 35, 26, 27, 30, 33]]
    fitnessCurve, best_fitness, efficiency = woa(channel_user, LB=1, UB=35,
                                     whale_num=whale_num, max_iter=iterators).opt()

    fig, ax1 = plt.subplots(figsize=(12, 10), dpi=300)
    plt.title('加入档案集后的网络传输速率和不满足传输速率', fontdict={'weight': 'normal', 'size': 30})
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
