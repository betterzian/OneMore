from src.scheduler.otherAlgorithm.randomFit import RandomFit
from src.scheduler.otherAlgorithm.firstFit import FirstFit
from src.scheduler.otherAlgorithm.bestFit import BestFit
from src.scheduler.otherAlgorithm.worstFit import WorstFit
from src.scheduler.otherAlgorithm.FGD import FGD
from src.scheduler.myAlgorithm.ResourceShaping import ResourceShaping
from src.scheduler.myAlgorithm.oneMore import OneMore
from src.scheduler.myAlgorithm.GGSV import GGSV
from src.envSim.simParam import ParamHolder


def init_scheduler(cluster, can_predict=False, ):
    if ParamHolder().filename[:5] == "param":
        return [GGSV(cluster)]
    if can_predict:
        can_predict = [True]
    else:
        can_predict = [False]
    schedulers = []
    for temp in can_predict:
        schedulers.append(ResourceShaping(cluster, temp))
        schedulers.append(OneMore(cluster, temp))
        schedulers.append(FGD(cluster, temp))
        schedulers.append(WorstFit(cluster, temp))
        schedulers.append(BestFit(cluster, temp))
        schedulers.append(FirstFit(cluster, temp))
        schedulers.append(RandomFit(cluster, temp))
    return schedulers
