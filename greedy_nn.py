import numpy as np
import pandas as pd

def GreedyNN(instance, seed = None, metric=None, **kwargs):
    '''
    kwargs can include free_wait and only_cost
    '''
    #n = instance.n
    #x = np.full(n, -1, dtype=int)
    sol, x = instance.build_nearest_neighbor(0, metric=metric, **kwargs)
    f = sol.f
    df = pd.DataFrame(dict(Fitness=[f], x=[ ' '.join(map(str,x[1:])) ], #(removed -1 for purposes of comparing poliastro to pykep results)
                           eval_ranks = [0], budget = [1], seed = [0],
                           metric = [metric], instance = [instance.instance_name],
                           Solver = ['GreedyNN']))
    return df

