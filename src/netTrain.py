import argparse
from src.scheduler.myAlgorithm.generateValue.generateContinuousValuebyMoE import run as runbyMoE
from src.scheduler.myAlgorithm.generateValue.generateContinuousValuebyMoE import run



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='param')
    parser.add_argument('-cuda', type=int, default=0)
    parser.add_argument('-filename', type=str, default="openb_pod_list_multigpu50")
    args = parser.parse_args()
    runbyMoE(args)
