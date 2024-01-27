
import time
from tqdm import tqdm
import pandas as pd
import os
from multiprocessing import Pool
import csv

pbar = tqdm()
# pbar = tqdm(total=4100000000)
# pbar.set_description('main')
update = lambda *args: pbar.update(1000000)


def process_csv(k, v):
    p = Pool(35)
    file = '/disk7T/vis/clusterdata/data/' + k + '.csv'
    filesize = os.stat(file).st_size
    lines = 100000
    csv_reader = pd.read_csv(file, nrows=lines + 1)
    csv_reader.iloc[0:1].to_csv('temp_test_1.csv')
    temp_file = 'temp_test_1.csv'
    tempfilesize1 = os.stat(temp_file).st_size
    csv_reader.iloc[0:lines + 1].to_csv('temp_test_2.csv')
    temp_file = 'temp_test_2.csv'
    tempfilesize2 = os.stat(temp_file).st_size
    raw_count = int(lines * 1.2 * (filesize - tempfilesize1) / (tempfilesize2 - tempfilesize1))
    pbar.total = raw_count
    pbar.set_description('main')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    csv_reader = pd.read_csv(file, iterator=True, chunksize=1000000, header=None, names=v)
    for chunk in csv_reader:
        p.apply_async(run, args=(chunk,), callback=update)  # 实例化进程对象
    p.close()
    p.join()
    print("分解完成")


def run(chunk):
    chunk.dropna(subset=['container_id'], inplace=True)
    container_dic = {}
    while (not chunk.empty):
        temp_container = chunk.iloc[0].at['container_id']
        container_dic[temp_container] = chunk[chunk['container_id'] == temp_container][['time_stamp', 'cpu_util_percent', 'mem_util_percent']]
        chunk = chunk[~(chunk['container_id'] == temp_container)]
    for key, value in container_dic.items():
        value.to_csv('/disk7T/vis/code/container/' + str(key) + '.txt', index=False, header=0, sep='\t',mode='a', encoding="utf-8", quoting=csv.QUOTE_NONE, escapechar=',')
    container_dic.clear()


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    dic = {
        #        'container_meta': ['container_id', 'machine_id', 'time_stamp', 'status', 'cpu_request', 'cpu_limit',
        #                           'mem_size'],
        'container_usage': ['container_id', 'machine_id', 'time_stamp', 'cpu_util_percent', 'mem_util_percent', 'cpi',
                            'mem_gps', 'mpki', 'net_in', 'net_out', 'disk_io_percent']}
    for k, v in dic.items():
        process_csv(k, v)

