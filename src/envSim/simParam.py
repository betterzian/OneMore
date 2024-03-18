from datetime import datetime
import json
import os

class SimParam:
    def __init__(self, args_list):
        if args_list is None:
            return
        current_time = datetime.now()
        self.zero = 0.000001
        self.time_len = args_list["tl"]
        self.time_init_flag = args_list["tif"]
        self.time_end_flag = args_list["tef"]
        self.time_can_predict = args_list["tcp"]
        self.time_block_size = args_list["tbs"]
        self.time_accurately_predict = args_list["tap"]
        self.online_task_num = args_list["ontn"]
        self.offline_task_num = args_list["oftn"]
        self.cpu_gpu_rate = self.__get_cgr(args_list["filename"])
        self.node_type = args_list["nt"]
        self.node_num = args_list["nn"]
        self.filename = args_list["filename"]
        self.device = args_list["device"]
        if not args_list["csv_name"]:
            self.csv_name = str(current_time.year) + "_" + str(current_time.month) + "_" + str(current_time.day) + "_" + str(current_time.hour) + "_" + str(current_time.minute) + "_" + str(current_time.second)
        else:
            self.csv_name = args_list["csv_name"]

    def init_again(self,args_list):
        self.__init__(args_list)

    def __get_cgr(self,filename):
        if os.path.exists("../srcData/state_value/" + filename + "/state_int.csv"):
            with open("../srcData/state_value/" + filename + "/data.json", "r") as json_file:
                data = json.load(json_file)
                return data["cgr"]
        else:
            return 0

class ParamHolder:
    _instance = None
    def __new__(cls, args_dict=None) -> SimParam:
        if cls._instance is None:
            cls._instance = SimParam(args_dict)
        return cls._instance
