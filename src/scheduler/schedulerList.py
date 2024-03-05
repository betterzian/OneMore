from src.scheduler.otherAlgorithm.randomFitScheduler import RandomFitScheduler
from src.scheduler.otherAlgorithm.firstFitScheduler import FirstFitScheduler
from src.scheduler.otherAlgorithm.bestFitScheduler import BestFitScheduler
from src.scheduler.otherAlgorithm.worstFitScheduler import WorstFitScheduler
from src.scheduler.myAlgorithm.varianceScheduler import VarianceScheduler
from src.scheduler.myAlgorithm.oneMoreScheduler import OneMoreScheduler
from src.scheduler.myAlgorithm.oneMoreModelScheduler import OneMoreModelScheduler
def init_scheduler(cluster):
    schedulers = []
    can_predict = [False,True,]
    for temp in can_predict:
        #schedulers.append(OneMoreScheduler(cluster,temp))
        schedulers.append(BestFitScheduler(cluster,temp))

        schedulers.append(OneMoreModelScheduler(cluster,temp))
        schedulers.append(VarianceScheduler(cluster,temp))
        schedulers.append(RandomFitScheduler(cluster,temp))
        schedulers.append(WorstFitScheduler(cluster,temp))
        schedulers.append(FirstFitScheduler(cluster, temp))
    return schedulers
