# 模拟程序
def sim_run(scheduler, online_task_list, offline_task_list,args_dict=None,count=None):
    from src.envSim.timeSim import TimeHolder
    from src.envSim.saveInfo import save_info
    from tqdm import tqdm
    from src.envSim.simParam import ParamHolder
    if args_dict is not None:
        ParamHolder().init_again(args_dict)
        TimeHolder().init_again()
    if type(scheduler).__name__ == "OneMore":
        scheduler.init_again()
    force_schedule = True
    if len(offline_task_list) == 0:
        force_schedule = False
    pbar = tqdm(total=TimeHolder().get_time_left(),
                desc=type(scheduler).__name__ + "," + str(scheduler.get_can_predict()))
    fail_task = []
    while online_task_list:
        now_task = online_task_list.pop()
        isOk = scheduler.run(now_task)
        if not isOk:
            fail_task.append(now_task)
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
            scheduler.run(now_task)
        TimeHolder().add_time()
        pbar.update(1)
    for node in scheduler.cluster:
        node.check(all_bool=True)
    save_info(scheduler)
    pbar.update(TimeHolder().get_time_left() - TimeHolder().get_fake_time_left())
    pbar.close()
    if type(scheduler).__name__ == "OneMoreModelScheduler":
        scheduler.release()
    return count

