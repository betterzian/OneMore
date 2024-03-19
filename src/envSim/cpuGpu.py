from src.envSim.timeBlockSim import TimeBlockSim
from src.envSim.timeSim import TimeHolder


class Cpu(TimeBlockSim):
    """
    资源块模拟类
    """
    def __init__(self, data,time_len=TimeHolder().get_time_len(),time_flag=TimeHolder().get_time_init_flag()):
        super().__init__(data, time_len, time_flag, is_cpu=True)


class Gpu(TimeBlockSim):
    def __init__(self, data,time_len=TimeHolder().get_time_len(),time_flag=TimeHolder().get_time_init_flag()):
        super().__init__(data, time_len, time_flag, is_cpu=False)
