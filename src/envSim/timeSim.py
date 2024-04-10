from src.envSim.simParam import ParamHolder
import multiprocessing
import threading


class TimeSim:
    """
    时间模拟器，使用单例模式，一个进程一个。
    其中self.__time_init_flag为模拟开始时间，在此时间之前的数据为历史数据，可供预测使用。
    """

    def __init__(self, wait=True):
        if wait:
            return
        self.__time_len = ParamHolder().time_len
        self.__time_init_flag = ParamHolder().time_init_flag
        self.__time_end_flag = ParamHolder().time_end_flag
        self.__time = 0
        self.__fake_time_left = self.__time_end_flag - self.__time_init_flag
        self.__time_left = self.__time_len - self.__time_init_flag

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

    def init_again(self, ):
        self.__init__(False)


class TimeHolder:
    _instances = {}

    def __new__(cls, wait=True, *args, **kwargs) -> TimeSim:
        process_id = multiprocessing.current_process().pid
        thread_id = threading.get_ident()
        if process_id not in cls._instances:
            cls._instances[process_id] = {}
        if thread_id not in cls._instances[process_id]:
            cls._instances[process_id][thread_id] = TimeSim(wait)
        return cls._instances[process_id][thread_id]
