import pandas as pd
import csv
from src.envSim.timeSim import TimeHolder
from envSim.generateEnv import generate_cluster, generate_online_task_list,generate_offline_task_list
from envSim.saveInfo import save_info
from scheduler.schedulerList import init_scheduler
from multiprocessing import Pool
from tqdm import tqdm
from datetime import datetime
import os

def run(scheduler, online_task_list, offline_task_list):
    scheduler.set_time()
    while online_task_list:
        now_task = online_task_list.pop()
        isOk = scheduler.run(now_task)
    save_info(scheduler)
    pbar = tqdm(total=TimeHolder().get_time_left(),desc=type(scheduler).__name__)
    fail_task = []
    reschedule_task = []
    while TimeHolder().get_time_left() > 0:
        for node in scheduler.cluster:
            reschedule_task.extend(node.check())
        scheduler.reschedule_num += len(reschedule_task)
        while reschedule_task:
            task = reschedule_task.pop()
            if task.get_arrive_time() < 0:
                online_task_list.append(task)
            else:
                offline_task_list.append(task)
        while fail_task:
            task = fail_task.pop()
            if task.get_arrive_time() < 0:
                online_task_list.append(task)
            else:
                offline_task_list.append(task)
        now_time = TimeHolder().get_time()
        while online_task_list:
            now_task = online_task_list.pop()
            isOk = scheduler.run(now_task)
        while offline_task_list:
            if offline_task_list[-1].get_arrive_time() <= now_time:
                now_task = offline_task_list.pop()
            else:
                break
            isOk = scheduler.run(now_task)
            if not isOk:
                fail_task.append(now_task)
        scheduler.fail_num += len(fail_task)
        TimeHolder().add_time()
        pbar.update(1)
    scheduler.set_time()
    save_info(scheduler)
    pbar.close()

def sim_run():
    cluster = generate_cluster(node_type=[(24,4),(48,8)],node_num=(72,24))
    #cluster = generate_cluster(node_type=[(24, 4), (48, 8)], node_num=(3, 3))
    online_task_list = generate_online_task_list()
    offline_task_list = generate_offline_task_list()
    schedulers = init_scheduler(cluster)
    p = Pool(len(schedulers))
    for scheduler in schedulers:
        p.apply_async(run, args=(scheduler, online_task_list, offline_task_list))
    p.close()
    p.join()
        #run(scheduler, online_task_list, offline_task_list)
    current_time = datetime.now()
    file = pd.read_csv("../output/scheduler_result.txt",header=None, sep='\t', encoding="utf-8", quoting=csv.QUOTE_NONE, escapechar=',')
    file.to_csv("../output/scheduler_result_"+str(current_time.year)+"_"+str(current_time.month)+"_"+str(current_time.day)+"_"+str(current_time.hour)+"_"+str(current_time.minute)+"_"+str(current_time.second)+".txt",index=False)
    os.remove("../output/scheduler_result.txt")