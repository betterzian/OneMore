import csv
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

update = lambda *args: pbar.update(1)


def run(file_name, cpu_request):
    chunk = pd.read_csv('../srcData/aliData/' + str(file_name) + '.txt', header=None, sep='\t',
                        encoding="utf-8", quoting=csv.QUOTE_NONE, escapechar=',',
                        names=['time_stamp', 'cpu_util_percent', 'mem_util_percent'])
    if chunk['cpu_util_percent'].sum() < 150000:
        return
    if chunk['mem_util_percent'].sum() < 150000:
        return
    min_time = chunk['time_stamp'].min()
    chunk['time_stamp'] = chunk['time_stamp'] - min_time
    assistance = np.array([i * 10 for i in range(69120)])
    assistance = pd.DataFrame(assistance, columns=['time_stamp'])
    chunk = pd.concat([chunk, assistance])
    chunk.drop_duplicates(subset=['time_stamp'], keep='first', inplace=True)
    chunk.sort_values(by='time_stamp', ascending=True, inplace=True)
    chunk['cpu_util_percent'].interpolate(method='linear', inplace=True)
    chunk['mem_util_percent'].interpolate(method='linear', inplace=True)
    chunk['cpu_util_percent'] = chunk['cpu_util_percent'] * cpu_request / 10000
    chunk = chunk['cpu_util_percent'].values[-17281:]
    chunk[0] = cpu_request / 100
    chunk.astype(np.float64)
    np.savetxt('../srcData/online_task/' + str(file_name) + '.csv', chunk, delimiter=",")
    return


# 处理分解后的container文件，数据清洗，插值，仅保留CPU利用率数据。
if __name__ == '__main__':
    meta = pd.read_csv('../srcData/aliData/container_meta.csv', header=None,
                       names=['container_id', 'machine_id', 'time_stamp', 'app_id', 'status', 'cpu_request',
                              'cpu_limit', 'mem_size'])
    meta = meta[meta['status'] == 'started'][['container_id', 'app_id', 'cpu_request']]
    meta.dropna(subset=['cpu_request'], inplace=True)
    meta.drop_duplicates(subset=['container_id', 'app_id'], keep='first', inplace=True)
    meta.sort_values(by='app_id', ascending=True, inplace=True)
    pbar = tqdm(total=21103)
    p = Pool(70)
    for row in meta.itertuples():
        p.apply_async(run, args=(row.container_id, row.cpu_request,), callback=update)
    p.close()
    p.join()
    pbar.close()
    chunk = pd.read_csv('../srcData/aliData/container.txt', header=None, sep='\t', encoding="utf-8",
                        quoting=csv.QUOTE_NONE, escapechar=',')
    chunk = chunk.values.astype(np.float64)
    np.savetxt('../srcData/aliData/container.csv', chunk, delimiter=",")
