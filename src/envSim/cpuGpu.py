from src.envSim.timeBlockSim import TimeBlockSim
from src.envSim.simParam import ParamHolder


class Cpu(TimeBlockSim):
    """
    资源块模拟类
    """

    def __init__(self, data, time_len=None, time_flag=None):
        super().__init__(data, time_len, time_flag, is_cpu=True)

    def set_info(self, data, start=-1):
        dic = {0: 0}
        data = data.reshape(1, -1)
        return self._set_info(data, start, dic)

    def get_info(self, length=-1):
        return self._get_info(length).reshape(-1)


class Gpu(TimeBlockSim):
    def __init__(self, data, time_len=None, time_flag=None):
        is_cpu = False
        if ParamHolder().filename[:5] == "param":
            is_cpu = True
        super().__init__(data, time_len, time_flag, is_cpu=is_cpu)

    def set_info(self, data, start=-1, dic={}):
        if len(dic) == 0:
            return
        return self._set_info(data, start, dic)

    def get_info(self, length=-1):
        return self._get_info(length)
