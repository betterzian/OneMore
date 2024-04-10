
import json
import os

class SimParam:
    def __init__(self, args_list):
        self.zero = 0.000001
        if args_list is None:
            return
        self.time_len = 2 if "tl" not in args_list.keys() else args_list["tl"]
        self.time_init_flag = 0 if "tif" not in args_list.keys() else args_list["tif"]
        self.time_end_flag = 1 if "tef" not in args_list.keys() else args_list["tef"]
        self.time_can_predict = 1 if "tcp" not in args_list.keys() else args_list["tcp"]
        self.time_block_size = 1 if "tbs" not in args_list.keys() else args_list["tbs"]
        self.time_accurately_predict = 1 if "tap" not in args_list.keys() else args_list["tap"]
        self.online_task_num = 1 if "ontn" not in args_list.keys() else args_list["ontn"]
        self.offline_task_num = 1 if "oftn" not in args_list.keys() else args_list["oftn"]
        self.node_type = ((32, 32), (96, 96)) if "nt" not in args_list.keys() else args_list["nt"]
        self.node_num = ((20, 20)) if "nn" not in args_list.keys() else args_list["nn"]
        self.filename = None if "filename" not in args_list.keys() else args_list["filename"]
        self.device = "cpu" if "device" not in args_list.keys() else args_list["device"]
        self.cuda = None if "cuda" not in args_list.keys() else args_list["cuda"]
        self.csv_name = None if "csv_name" not in args_list.keys() else args_list["csv_name"]
        self.avgsize = None if "avgsize" not in args_list.keys() else args_list["avgsize"]
        self.prob = 4 if "prob" not in args_list.keys() else args_list["prob"]
        self.weight_rate = 1 if "weight_rate" not in args_list.keys() else args_list["weight_rate"]
        self.zero_task = self.__get_zero_task(args_list["filename"]) if "zero_task" not in args_list.keys() else args_list["zero_task"]
        self.weight = self.weight_rate * self.avgsize
        self.all_node_cpu = 0
        self.all_node_gpu = 0
        self.__get_all_node_info()
        self.cpu_gpu_rate = self.__get_cgr(args_list["filename"]) if "cgr" not in args_list.keys() else args_list["cgr"]




    def init_again(self,args_list):
        self.__init__(args_list)

    def __get_all_node_info(self):
        self.all_node_cpu = 0
        for i in range(len(self.node_type)):
            self.all_node_cpu += self.node_type[i][0] * self.node_num[i]
            self.all_node_gpu += self.node_type[i][1] * self.node_num[i]


    def __get_cgr(self,filename):
        if os.path.exists("../srcData/state_value/" + filename + "/data.json"):
            with open("../srcData/state_value/" + filename + "/data.json", "r") as json_file:
                data = json.load(json_file)
                return data["cgr"]
        else:
            return 0

    def __get_zero_task(self,filename):
        if os.path.exists("../srcData/state_value/" + filename + "/data.json"):
            with open("../srcData/state_value/" + filename + "/data.json", "r") as json_file:
                data = json.load(json_file)
                return data["zero_task"]
        else:
            return 0



class ParamHolder:
    _instance = None
    def __new__(cls, args_dict=None) -> SimParam:
        if cls._instance is None:
            cls._instance = SimParam(args_dict)
        return cls._instance
