from src.envSim.timeBlockSim import TimeBlockSim
import numpy as np
from src.envSim.timeSim import TimeHolder
class CpuGpu(TimeBlockSim):
    """
    资源块模拟类
    """
    def __init__(self,data,time_len=TimeHolder().get_time_len(),time_flag=TimeHolder().get_time_init_flag()):
        self.__time_len = time_len
        self.__time_flag = time_flag
        if isinstance(data, (int, float, complex)):
            self.__data = np.ones(self.__time_len) * data
        else:
            self.__data = np.array(data)

    def set_info(self,data,start = -1):
        if start == -1:
            start = TimeHolder().get_time() + self.__time_flag
        data = np.array(data)
        data_len = len(data)
        self.__data = np.concatenate((self.__data[:start], self.__data[start:start + data_len] - data, self.__data[start + data_len:]), axis=0)

    def get_info(self,len = -1):
        if len < 0:#返回所有资源信息
            return self.__data[self.__time_flag:]
        temp_time_flag = TimeHolder().get_time() + self.__time_flag
        return self.__data[temp_time_flag:temp_time_flag + len].copy() #返回当前时刻后的资源

    def check(self):
        time = TimeHolder().get_time() + TimeHolder().get_time_init_flag()
        return self.__data[time]