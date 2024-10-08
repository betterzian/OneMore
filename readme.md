# Hardware

Specify the hardware requirements and dependencies (e.g., a specific interconnect or GPU type is required).

The simulation experiments are conducted on Intel Xeon(R) Gold 6430 CPU @2.10 GHz$\times$64 with 480 GBytes of RAM and four GeForce RTX 4090s, each with 24 GBytes of memory. The configuration, whether higher or lower, should only influence the simulation time, not the outcomes.

# Software

Introduce all required software packages, including the computational artifact. For each software package, specify the version and provide the URL.

These packages are utilized: python==3.9.18, numpy==1.23.5, pandas==2.1.4, torch==2.1.2, tqdm==4.65.0, along with cuda==12.1.0. For data analysis and graphing, Statgraphics 19 is needed, which is available at https://www.statgraphics.com/download19.

# Datasets / Inputs

Describe the datasets required by the artifact. Indicate whether the datasets can be generated, including instructions, or if they are available for download, providing the corresponding URL.

The datasets used in the paper are the Alibaba 2018 cluster trace dataset and the Alibaba 2023 cluster trace dataset, available at https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018 and https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2023, respectively. The dataset for parameter calibration is generated randomly.

Processing the 2018 dataset files requires about 200 CPU hours. The data has been cleaned and provided in a zip file; once unzipped, it occupies approximately 6G of memory. The 2023 dataset is also available locally, with no modifications.

# Installation and Deployment

Detail the requirements for compiling, deploying, and executing the experiments, including necessary compilers and their versions.

Here are the details of using the simulator on Ubuntu 22.04. It can also operate on Windows or macOS with adjustments to the respective commands.

## Install OneMore

```bash
git clone https://github.com/betterzian/OneMore.git
cd OneMore
```

## Deployment OneMore

```bash
cd OneMore
pip3 install -r requirements.txt
cd srcData
unzip online_task.zip online_task
```

To replicate the process of generating data from the Alibaba 2018 cluster trace dataset, first download 	the files **container\_usage.csv** and **container\_meta.csv** into the **aliDataProcess** folder. Then, follow 	these steps:

```bash
cd OneMore/aliDataProcess
python ali2018Container.py
python containerCleaning.py
```

# Artifact Execution

## 1. Conducting a Single Experiment with a Set of Parameters

```bash
cd OneMore/src
python sinRunMain.py 
```

Each simulator instance includes a main scenario in the paper that operates without resource prediction. When running **sinRunMain.py**, a few optional parameters can be adjusted, as detailed in the table below:

| Parameters |             Meanings              | Default |
| :--------: | :-------------------------------: | :-----: |
|   -cuda    |    number of GPU, -1 means CPU    |    0    |
|     -t     |        simulation duration        |  8640   |
|    -tcp    |   duration of trend prediction    |  8640   |
|    -tap    |   duration of value prediction    |   30    |
|   -ontn    |      number of online tasks       |  1000   |
|   -oftn    |      number of offline tasks      |  1400   |
| -filename  | the name of offline tasks dataset |  None   |

For instance, to deploy a single experiment with 2000 online tasks using only the CPU, the command is:

```bash
python sinRunMain.py -mc=-1 -ontn=2000
```

## 2. Conducting Numerous Experiments with Different Sets of Parameters

```bash
cd OneMore/src
# for algorithm comparison
python multiRunMain.py 
# for parameter calibration
python multiRunMain.py -param 
```

The table below outlines the optional parameters that can be specified when executing **multiRunMain.py**:

| Parameters |            Meanings            | Default |
| :--------: | :----------------------------: | :-----: |
|   -pool    |     size of threading pool     |   64    |
|    -mc     |         number of GPUs         |    4    |
|    -ins    | number of repeated experiments |    5    |

The table below presents the parameters that can be modified for the experiment within the **config.py** file. These include **args\_dict\_param** for parameter calibration and **args\_dict\_compare** for algorithm comparison. The description of these parameters is detailed in the table:

|    Parameters     |              meanings              |
| :---------------: | :--------------------------------: |
|    prob\_list     |             prob value             |
|    size\_list     |       biggest size of tasks        |
| small\_task\_list |       smallest size of tasks       |
|  task\_num\_list  |          number of tasks           |
| node\_count\_list |          number of nodes           |
|     tcp\_list     |    duration of trend prediction    |
|     tap\_list     |    duration of value prediction    |
| oftn\_ontn\_list  | number of offline and online tasks |
|  filename\_list   |    name of offline task dataset    |

To train the network responsible for obtaining state values, execute the following commands:

```bash
cd OneMore/src
python netTrain.py
```

The table below outlines the optional parameters that can be adjusted when executing the **netTrain.py**:

| Parameters |             meanings              | default |
| :--------: | :-------------------------------: | :-----: |
| -filename  | the name of offline tasks dataset |  None   |
|   -cuda    |    number of GPU, -1 means CPU    |    0    |

The models utilized in this study are accessible from the folder  **OneMore/srcData/offline\_task/model** and ready for immediate use.

To replicate the experiment and compare the algorithms as soon as possible, please follow these steps using the resources we have prepared:

```bash
cd OneMore/src
# for parameter calibration
python multiRunMain.py -param
# When done, do not change the file config, for algorithm comparison.
python multiRunMain.py -pool=64 -mc=4 -ins=5
```

## 3. main processes

The essential code for constructing the main environment for the simulator can be found in the folder **OneMore/src/envSim**, while the code for the scheduler is situated in **OneMore/src/scheduler**. To integrate a new scheduling algorithm, simply develop a new scheduler class within **OneMore/src/scheduler/otherAlgorithm**. Below is a provided example for guidance:

```python
from src.scheduler.schedulerClass import Scheduler
class Example(Scheduler):
    def __init__(self,cluster,can_predict = True,task_mem = {},node_mem = {}):
        super().__init__(cluster,can_predict,task_mem,node_mem)
        #The function to schedule a task
def run(self,task): 
    task_cpu, task_gpu = self.get_task_info(task) #get task information
    now_select = -1 # selected node
    gpu_select = {} # selected GPU
    for node in self.cluster:
        temp_select = {}
        temp_node_cpu, temp_node_gpu = self.get_node_info(node) #get node information
        # algorithm
        # algorithm
        # algorithm
    if now_select != -1:
        self.set_task(now_select, task, gpu_select) #run a task in a node
        return True
    else:
        return False
```

Then incorporate the scheduler by importing it into the file **OneMore/src/scheduler/schedulerList.py**.

# Analysis

The experiment results, parameter calibration, and algorithm comparison data generated by the simulator are saved in specific CSV files within the **OneMore/output** directory with their filenames containing a timestamp placeholder represented by {timeflag}. Here’s how the simulator handles the output files:

For a single experiment, the result is recorded in the file named **scheduler\_result\_\{timeflag\}.csv**.

When performing parameter calibration, the outcomes are documented in the file entitled **scheduler\_result\_all\_param\_\{timeflag\}.csv**.

The results of comparisons between different scheduling algorithms, through multiple experiments, are stored in the file **scheduler\_result\_all\_compare\_\{timeflag\}.csv**.

After the simulation runs, these files can be used to conduct a multifactorial analysis with a software tool like Statgraphics 19.

