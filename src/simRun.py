def run(scheduler, online_task_list, offline_task_list):
    from src.envSim.timeSim import TimeHolder
    from envSim.saveInfo import save_info
    from tqdm import tqdm
    from src.envSim.simParam import ParamHolder
    force_schedule = True
    if len(offline_task_list) == 0:
        force_schedule = False
    scheduler.set_time()
    fail_task = []
    while online_task_list:
        now_task = online_task_list.pop()
        isOk = scheduler.run(now_task)
        if not isOk:
            fail_task.append(now_task)
    #save_info(scheduler)
    pbar = tqdm(total=TimeHolder().get_time_left(),
                desc=type(scheduler).__name__ + "," + str(scheduler.get_can_predict()))
    reschedule_task = []
    while TimeHolder().get_fake_time_left() > 0:
        for node in scheduler.cluster:
            reschedule_task.extend(node.check())
        scheduler.add_reschedule_num(len(reschedule_task))
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
            if force_schedule and not isOk:
                reschedule_task.extend(scheduler.force_set_online_task(now_task))
        scheduler.add_reschedule_num(len(reschedule_task))
        while reschedule_task:
            task = reschedule_task.pop()
            if task.get_arrive_time() < 0:
                return
            else:
                offline_task_list.append(task)
        while offline_task_list:
            if offline_task_list[-1].get_arrive_time() <= now_time:
                now_task = offline_task_list.pop()
            else:
                break
            isOk = scheduler.run(now_task)
        #     if not isOk:
        #         fail_task.append(now_task)
        # scheduler.add_fail_num(len(fail_task))
        TimeHolder().add_time()
        pbar.update(1)
    scheduler.set_time()
    while TimeHolder().get_time_left() > 0:
        for node in scheduler.cluster:
            node.check()
        TimeHolder().add_time()
        pbar.update(1)
    save_info(scheduler)
    pbar.close()
    with open('../tmp/'+ParamHolder().csv_name+'.txt', 'a') as file:
        file.write('ok\n')

def sim_run(args_dict):
    from src.envSim.simParam import ParamHolder
    ParamHolder(args_dict)
    file = open('../tmp/all_'+ParamHolder().csv_name+'.txt', "w")
    file.close()
    from src.envSim.offlineTaskDataProcess import offline_data_process
    offline_data_process(ParamHolder().filename)
    args_dict["cgr"] = ParamHolder().cpu_gpu_rate
    args_dict["csv_name"] = ParamHolder().csv_name
    from src.envSim.simParam import ParamHolder
    from envSim.generateEnv import generate_cluster, generate_online_task_list, generate_offline_task_list
    from scheduler.schedulerList import init_scheduler
    from multiprocessing import Pool
    from src.envSim.TXTtoCSV import txt_to_csv
    cluster = generate_cluster()
    online_task_list = generate_online_task_list()
    offline_task_list = generate_offline_task_list()
    schedulers = init_scheduler(cluster)
    if not args_dict["test"]:
        p = Pool(len(schedulers))
        for scheduler in schedulers:
            p.apply_async(run, args=(scheduler, online_task_list, offline_task_list))
        p.close()
        p.join()
    else:
        for scheduler in schedulers:
            print("单进程,测试用")
            run(scheduler, online_task_list, offline_task_list)
    if ParamHolder().csv_name[:3] != "all":
        txt_to_csv(ParamHolder().csv_name)
