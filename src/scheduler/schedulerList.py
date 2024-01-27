from src.scheduler.smallGreedyScheduler import SmallGreedyScheduler
from src.scheduler.largeGreedyScheduler import LargeGreedyScheduler
from src.scheduler.normalScheduler import NormalScheduler
from src.scheduler.randomFitScheduler import RandomFitScheduler
from src.scheduler.myAlgorithm.varianceScheduler import VarianceScheduler

def init_scheduler(cluster):
    schedulers = []
    #schedulers.append(SmallGreedyScheduler(cluster))
    #schedulers.append(LargeGreedyScheduler(cluster))
    #schedulers.append(NormalScheduler(cluster))
    schedulers.append(RandomFitScheduler(cluster))
    #schedulers.append(VarianceScheduler(cluster))
    return schedulers