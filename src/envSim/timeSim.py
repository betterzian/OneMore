from src.simParam import __time_len__,__time_init_flag__,__time_end_flag__
import multiprocessing
import threading
class TimeSim:
    """
    时间模拟器，使用单例模式，一个进程一个。
    其中self.__time_init_flag为模拟开始时间，在此时间之前的数据为历史数据，可供预测使用。
    """
    def __init__(self,time_len,time_init_flag,time_end_flag):
        self.__time_len = time_len
        self.__time_init_flag = time_init_flag
        self.__time_end_flag = time_end_flag
        self.__time = 0
        self.__time_left = self.__time_end_flag - self.__time_init_flag
        self.__fake_time_left = self.__time_len - self.__time_init_flag

    def add_time(self):
        self.__time += 1
        self.__time_left -= 1
        self.__fake_time_left -= 1

    def get_time_len(self):
        return self.__time_len

    def get_time_left(self):
        return self.__time_left

    def get_time_init_flag(self):
        return self.__time_init_flag

    def get_time_end_flag(self):
        return self.__time_end_flag

    def get_fake_time_left(self):
        return self.__fake_time_left

    def get_time(self):
        return self.__time


class TimeHolder:
    _instances = {}
    def __new__(cls, *args, **kwargs) -> TimeSim:
        process_id = multiprocessing.current_process().pid
        thread_id = threading.get_ident()
        if process_id not in cls._instances:
            cls._instances[process_id] = {}
        if thread_id not in cls._instances[process_id]:
            cls._instances[process_id][thread_id] = TimeSim(__time_len__,__time_init_flag__,__time_end_flag__)
        return cls._instances[process_id][thread_id]