import math
import sys

import numpy as np
from numpy import random

# 阶段1 完成单个用户情况下路径损耗和噪声公式的传输速率
# 香农定理  C=B * log2(1+SINR)  c单位 bps 比特每秒   B 信道带宽   hz  SNR 信噪比  1Mbps = 1024*1024bps 1Mbps=128KB/s
# SINR = 10 * log10(信号功率/（噪声功率+干扰功率））
# SNR(dB) = 信号功率(dBm) - 路径损失(dB) - 噪声功率(dBm)
# 15KHz的噪声功率 Noise Power (dBm)= -174 +10*log(Bandwidth in Hz) = -174 + 10*log(15*1000) =-132.23 dBm
# 简单的能量消耗模型  E = P * t  p发射功率，通常以W为单位  t传输速率，单位比特每秒（bps）或字节每秒（Bps）
# 通常手机信号满格时手机的辐射功率为10-20dBm（0.01-0.1W）左右，
# 信号很差时，辐射功率会达到最高33dBm（2W）
# 20W->dBm  p(dBm) = 10 * log10(1000*20W) = 43.0103dBm
# 工信部2021-10-13发布通过 非授权频段，2.4GHz  20MHz带宽  2400MHz频率  发射功率大于20dBm   14个信道
#                                 5GHz   160MHz带宽  5200MHz频率 发射功率大于30dBm

# 计算噪声功率,采用白噪声密度 -174dBm/Hz
# 以WCDMA为例。终端发射功率最大为125mW（20.97dBm），而基站接收灵敏度为-124.67dBm

c = 3e+08


# 信道增益 = - 路径损耗 -阴影衰落 + 天线增益
def cal_link_gain(distance):
    pl_ci = 128 + 37.6 * math.log10(distance / 1000)
    return pl_ci


def cal_signal_noise(distance, station_index, firing_power, carrier_frequency, occupancy_period):
    """
    计算MBs下的干扰信噪比
    :param distance:
    :param user: 用户信息
    :param i: 该信道下第i个用户
    :param firing_power: 用户i在该信道的发射功率
    :param carrier_frequency: 基站的载波频率
    :return:
    """
    other_station_index = 1 if station_index == 0 else 0
    power_gain = np.zeros(len(occupancy_period[station_index]))
    for second_num in range(1, 101):
        station_user_index = 0
        other_station_user_index = 0
        for user_index in range(len(occupancy_period[station_index])):
            if second_num <= occupancy_period[station_index][user_index]:
                station_user_index = user_index
                break
        for other_user_index in range(len(occupancy_period[other_station_index])):
            if second_num <= occupancy_period[other_station_index][other_user_index]:
                other_station_user_index = other_user_index
                break
        power_gain[station_user_index] += math.pow(10, cal_link_gain(distance[other_station_index][other_station_user_index]) / 10) * firing_power[other_station_index][other_station_user_index]
    sinr = []
    for user_index in range(len(firing_power[station_index])):
        link_gain = math.pow(10, (cal_link_gain(distance[station_index][user_index])) / 10) * firing_power[station_index][user_index]
        if power_gain[user_index] == 0.0:
            sinr.append(0)
        else:
            sinr.append(link_gain / (power_gain[user_index] + math.pow(10, (-174+10*math.log10(carrier_frequency))/10)))
    return sinr


def cal_shannon_rate(sinr, bandwidth, channel):
    authorized_rate = []
    for i in range(len(sinr)):
        authorized_rate.append((channel[i] / 100) * bandwidth * np.log2(1 + sinr[i]))
    return authorized_rate


def cal_rate(channel_user, distances):
    occupancy_period = np.zeros([len(channel_user), len(channel_user[0])])
    # 遍历信道，修改每个信道上用户的连接时间
    for station_index in range(len(channel_user)):
        for user_index in range(len(channel_user[station_index])):
            current_occupancy = channel_user[station_index][user_index]
            for index in range(1, user_index + 1):
                current_occupancy += channel_user[station_index][index]
            occupancy_period[station_index][user_index] = current_occupancy / sum(channel_user[station_index]) * 100
    carrier_frequency = 2400e+06  # 载波频率2400MHz，单位Hz
    bandwidth = 20e+06  # 授权频段每个信道的带宽

    # 按照基站遍历
    user_rate = []  # 所有用户在该信道的总传输速率
    for i in range(len(channel_user)):
        interference_sinr = []  # 信道的干扰信噪比
        # 计算基站传输速率
        user_i_signal_noise = cal_signal_noise(distances, i, channel_user, carrier_frequency, occupancy_period)
        rate = cal_shannon_rate(user_i_signal_noise, bandwidth, channel_user[i])
        user_rate.append(np.sum(rate))

    return sum(user_rate)


def cal_rate_feasible(channel_user, distances):
    occupancy_period = np.zeros([len(channel_user), len(channel_user[0])])
    # 遍历信道，修改每个信道上用户的连接时间
    for station_index in range(len(channel_user)):
        for user_index in range(len(channel_user[station_index])):
            current_occupancy = channel_user[station_index][user_index]
            for index in range(1, user_index + 1):
                current_occupancy += channel_user[station_index][index]
            occupancy_period[station_index][user_index] = current_occupancy / sum(channel_user[station_index]) * 100
    # 系统模型中1个MBS和两个FBS FBS一边公用MBs用户的上行链路传输自己的数据，一连使用LBT机制分享wifi的非授权资源 24个信道
    carrier_frequency = 2400e+06  # 载波频率2400MHz，单位Hz
    bandwidth = 20e+06  # 授权频段每个信道的带宽

    # 按照基站遍历
    user_rate = []  # 所有用户在该信道的总传输速率
    inequality_rate = []
    feasible_count = []
    for i in range(len(channel_user)):
        feasible_count_temp = []
        # 计算基站传输速率
        user_i_signal_noise = cal_signal_noise(distances, i, channel_user, carrier_frequency, occupancy_period)
        rate = cal_shannon_rate(user_i_signal_noise, bandwidth, channel_user[i])
        # print(rate)
        for j in range(0, len(channel_user[i])):
            # 如果低于5E+06，说明不符合条件
            inequality_rate.append(max(0, 1E+05 - rate[j]))
            feasible_count_temp.append(inequality_rate[j])
        user_rate.append(rate)
        feasible_count.append(feasible_count_temp)
    return user_rate, feasible_count
