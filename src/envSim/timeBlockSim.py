import math
import numpy as np
from src.envSim.timeSim import TimeHolder


class TimeBlockSim:
    def __init__(self, data, time_len, time_flag, is_cpu):
        if time_len is None:
            time_len = TimeHolder().get_time_len()
        if time_flag is None:
            time_flag = TimeHolder().get_time_init_flag()
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

    def _set_info(self, data, start, dic):
        if start == -1:
            start = TimeHolder().get_time() + self.__time_flag
        data_len = len(data[0])
        self.__data[np.array(list(dic.keys())), start:start + data_len] = np.round(self.__data[np.array(list(dic.keys())), start:start + data_len] - data[np.array(list(dic.values()))], 1)

    def _get_info(self, length):
        if length < 0:  # 返回所有资源信息
            return self.__data[:,self.__time_flag:].copy()
        temp_time_flag = TimeHolder().get_time() + self.__time_flag
        return self.__data[:,temp_time_flag:temp_time_flag + length].copy()  # 返回当前时刻后的资源

    def check(self, all_bool):
        time = TimeHolder().get_time() + TimeHolder().get_time_init_flag()
        if all_bool:
            return self.__data[:, time:].min() < 0
        return self.__data[:, time].min() < 0
