import math

from src.envSim.timeSim import TimeHolder
import numpy as np


class TimeBlockSim:
    def __init__(self, data, time_len, time_flag, is_cpu):
        self.__time_len = time_len
        self.__time_flag = time_flag
        if is_cpu:
            self.__length = 1
            self.__is_cpu = True
            if isinstance(data, (int, float, complex)):
                self.__data = np.ones((1, self.__time_len)) * data
            else:
                self.__data = np.array(data).reshape(1, -1)
        else:
            self.__is_cpu = False
            self.__length = math.ceil(data)  # GPU资源data为数字
            if data == 0:
                self.__data = np.array([[]])
            if data < 1:
                self.__data = np.ones((1, self.__time_len)) * data
            else:
                self.__data = np.ones((self.__length, self.__time_len))

    def set_info(self, data, start=-1, dic=None):
        if dic is None and self.__is_cpu:
            dic = {0: 0}
        if start == -1:
            start = TimeHolder().get_time() + self.__time_flag
        if self.__data:
            data_len = len(data[1])
            self.__data[np.array(list(dic.key())), start:start + data_len] = np.round(self.__data[np.array(list(dic.key())), start:start + data_len] - data[np.array(list(dic.values()))], 1)

    def get_info(self, len=-1):
        if len < 0:  # 返回所有资源信息
            return self.__data[:,self.__time_flag:].copy()
        temp_time_flag = TimeHolder().get_time() + self.__time_flag
        return self.__data[:,temp_time_flag:temp_time_flag + len].copy()  # 返回当前时刻后的资源

    def check(self):
        time = TimeHolder().get_time() + TimeHolder().get_time_init_flag()
        return np.any(self.__data[:,time] < 0)
