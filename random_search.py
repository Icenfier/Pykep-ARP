import numpy as np
import pandas as pd

def RandomSearch(instance, seed, budget):
    #np.random.seed(seed)
    seeds = [1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3, 30, 4, 5, 6, 7, 8, 9]
    n = instance.n
    sample = []
    fitnesses = []
    evaluations = []
    seeds_used = []
    for sd in seeds:
        np.random.seed(sd)
        for m in range(budget):
            p = np.random.permutation(n)
            f = instance.fitness(p)
            sample.append(p)
            fitnesses.append(f)
            evaluations.append(m+1)
            seeds.append(sd)
    df = pd.DataFrame()
    df['Fitness'] = fitnesses
    df['x'] = [ ' '.join(map(str,s)) for s in sample ]
    df['seed'] = seeds_used
    df['budget'] = budget
    df['Function evaluations'] = evaluations
    df['instance'] = [instance.instance_name] * len(evaluations)
    df['Solver'] = ['RandomSearch'] * len(evaluations)
    return df
