import argparse

def generate_sim_param():
    with open("simParam.py", "w") as f:
        f.write(f"__time_len__ = {8640}\n")
        f.write(f"__time_init_flag__ = {0}\n")
        f.write(f"__time_can_predict__ = {7200}\n")
        f.write(f"__time_block_size__ = {90}\n")
        f.write(f"__time_accurately_predict__ = {90}\n")
        f.write(f"__online_task_num__ = {1000}\n")
        f.write(f"__offline_task_num__ = {10000}\n")
        f.write(f"__zero__ = {0.0001}\n")
        f.write(f"__cpu_gpu_rate__ = {4}\n")

if __name__ == '__main__':
    generate_sim_param()
    from simRun import sim_run
    sim_run()





