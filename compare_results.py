from arp import AsteroidRoutingProblem
from pandas import DataFrame, read_csv, concat
import os
import numpy as np
from greedy_nn import GreedyNN
from random_search import RandomSearch

"""
Reads in pre-existing results from when this code relied on poliastro. 
Performs the same calculations for comparison.
"""

#### GreedyNN ####
'''
blank = DataFrame(data=[[0,'#','#','#','#','#','#','#']], 
                     columns=['Fitness','x','eval_ranks','budget','seed','metric','instance','Solver'])
to_merge = [blank]
for folder in os.listdir('D:/UniCompSci/ARP internship/GitHub/Pykep-ARP/results/greedynn'):
    inst = folder.split('_')
    arp_instance = AsteroidRoutingProblem(int(inst[1]), int(inst[2]))
    for file in os.listdir(f'D:/UniCompSci/ARP internship/GitHub/Pykep-ARP/results/greedynn/{folder}'):
        poli_data = read_csv(f'D:/UniCompSci/ARP internship/GitHub/Pykep-ARP/results/greedynn/{folder}/{file}', 
                             header=0, sep=',', index_col=False,
                        names=['Fitness','x','eval_ranks','budget','seed','metric','Function evaluations','run_time','Problem','instance','Solver'])
        poli_data.drop(['Function evaluations','run_time','Problem'],  axis=1, inplace=True)
        metric = poli_data.iloc[0,5] # index corresponds to metric
        pk_data = GreedyNN(arp_instance, metric=metric)
        
        to_merge.extend([poli_data, pk_data, blank])

greedynn_combined = concat(to_merge)        
print(greedynn_combined)
'''
#### RandomSearch ####
'''
rand_comparisons = {}
for folder in os.listdir('D:/UniCompSci/ARP internship/GitHub/Pykep-ARP/results/randomsearch'):
    inst = folder.split('_')
    arp_instance = AsteroidRoutingProblem(int(inst[1]), int(inst[2]))
    seed = 1
    budget = 400 

    for file in os.listdir(f'D:/UniCompSci/ARP internship/GitHub/Pykep-ARP/results/randomsearch/{folder}'):
        poli_data = read_csv(f'D:/UniCompSci/ARP internship/GitHub/Pykep-ARP/results/randomsearch/{folder}/{file}', 
                             header=0, sep=',', index_col=False,
                        names=['Fitness','x','seed','budget','Function evaluations','run_time','Problem','instance','Solver'])
        poli_data.drop(['run_time','Problem'],  axis=1, inplace=True)
        pk_data = RandomSearch(arp_instance, seed, budget)
        
        dfs = [poli_data, pk_data]
        idx = np.argsort(np.concatenate([np.arange(d.shape[0]) for d in dfs]))
        rand_comparisons["instance{0}".format(arp_instance.instance_name)] = concat(dfs, ignore_index=True).iloc[idx]
        
print(rand_comparisons)
'''