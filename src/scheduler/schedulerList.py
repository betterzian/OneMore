from src.scheduler.otherAlgorithm.randomFitScheduler import RandomFitScheduler


def init_scheduler(cluster):
    schedulers = []
    #schedulers.append(SmallGreedyScheduler(cluster))
    #schedulers.append(LargeGreedyScheduler(cluster))
    #schedulers.append(NormalScheduler(cluster))
    schedulers.append(RandomFitScheduler(cluster))
    #schedulers.append(VarianceScheduler(cluster))
    return schedulers
