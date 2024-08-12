from arp import AsteroidRoutingProblem
from arp_vis import plot_solution
import numpy as np

""" 
plot_solution(instance, sequence, solution)
                Plots a trajectory for visiting a given set of asteroids. A
                solution object can be provided to avoid re-evaluation. If no 
                solution object is provided, the sequence will be re-evaluated 
                before plotting.
                
arp_instance._______
            .CompleteSolution(sequence)
                Returns a Solution object for the ARP from a given sequence of 
                asteroids.
                
            .evaluate_sequence(sequence)
                Returns objective function value and wait/transfer times for a
                given sequence of asteroids.
            
            .optimize_transfer(from_id, to_id, current_time)
                Returns a Solution object for a single transfer orbit.
                
            .build_nearest_neighbor(current_time, metric, use_phasing)
                Builds a 'greedy' nearest neighbour solution. 
                Metric can be 'euclidean' or 'orbital', corresponding to nearest 
                neighbour by physical distance or orbital energy. 
                Use_phasing, when true, uses orbital phasing to find nearest 
                neighbours, also using either 'euclidean' distance or 
                'orbital' energy.
                
            .evaluate_transfer(from_id, to_id, start_time, t0, t1)
                Returns objective function value and maneuver of a single
                Lambert transfer, given parameters for wait transfer times.
                
            
Additional arguments:
    free_wait - If true, wait times on asteroids aren't included in the 
                objective function.
    only_cost - If true, time is not counted, only the cost of changing 
                velocities.
            

"""

### Plot projection, either '2d' or '3d'.  Only use 3d if figure can be    ###
### opened in a new window (for example, using '%matplotlib qt' in Spyder) ###
projection = '2d' 

### Asteroid Routing Problem parameters
n = 10    # No. of asteroids
seed = 42 # Seed for random asteroid selector

### Initialise instance ###
arp_instance = AsteroidRoutingProblem(n, seed)



x = np.arange(0,n)
np.random.shuffle(x)
res1 = arp_instance.CompleteSolution(np.asarray(x))
res2 = arp_instance.evaluate_sequence([-1] + x, current_time=0)
plot_solution(arp_instance, x, res1, projection = projection)
print(res2)


solution = arp_instance.optimize_transfer(0, 8, current_time = 0, t0_bounds = (0, 5110), t1_bounds = (1, 730), free_wait = True, multi = 3)
print(solution.f)  

solution = arp_instance.optimize_transfer(0, 8, current_time = 126.96358, t0_bounds = (0,4983.0366), t1_bounds = (1, 730), free_wait = True, multi = 3)
print(solution.f)        

solution = arp_instance.optimize_transfer(0, 3, current_time = 0, t0_bounds = (0, 13870), t1_bounds = (1, 730), free_wait = True, multi=3)
print(solution.f)        

solution = arp_instance.optimize_transfer(0, 3, current_time = 69.746826, t0_bounds = (0,2120.2532), t1_bounds = (1, 730), free_wait = True, multi=4)
print(solution.f)        


# Build nearest neighbor solutions

# Euclidean
sol, sequence = arp_instance.build_nearest_neighbor(current_time = 0, use_phasing = False)
print(f"*** sequence = {sequence}, t = {sol.ship.leg_times}, cost = {sol.get_cost()}\n\n")
plot_solution(arp_instance, sequence[1::], sol, projection = projection)  

sol, sequence = arp_instance.build_nearest_neighbor(current_time = 0, free_wait = True, use_phasing = False)
print(f"*** sequence = {sequence}, t = {sol.ship.leg_times}, cost = {sol.get_cost()}\n\n")
plot_solution(arp_instance, sequence[1::], sol, projection = projection)  

sol, sequence = arp_instance.build_nearest_neighbor(current_time = 0, only_cost = True, use_phasing = False)
print(f"*** sequence = {sequence}, t = {sol.ship.leg_times}, cost = {sol.get_cost()}\n\n")
plot_solution(arp_instance, sequence[1::],  sol, projection = projection)

sol, sequence = arp_instance.build_nearest_neighbor(current_time = 0, use_phasing = True)
print(f"*** sequence = {sequence}, t = {sol.ship.leg_times}, cost = {sol.get_cost()}\n\n")
plot_solution(arp_instance, sequence[1::], sol, projection = projection) 


# Orbital
sol, sequence = arp_instance.build_nearest_neighbor(current_time = 0, metric='orbital')
print(f"*** sequence = {sequence}, t = {sol.ship.leg_times}, cost = {sol.get_cost()}\n\n")
plot_solution(arp_instance, sequence[1::],  sol, projection = projection)  

sol, sequence = arp_instance.build_nearest_neighbor(current_time = 0, metric='orbital', only_cost = True, use_phasing = False)
print(f"*** sequence = {sequence}, t = {sol.ship.leg_times}, cost = {sol.get_cost()}\n\n")
plot_solution(arp_instance, sequence[1::],  sol, projection = projection)  




# Individual transfer orbits
solution = arp_instance.optimize_transfer(1, 2, current_time = 574, t0_bounds = (0, 1), t1_bounds = (240, 260))
print(solution)        

solution = arp_instance.optimize_transfer(1, 2, current_time = 574, total_time_bounds = (200,260))
print(solution)        



from_id = -1 # From Earth
to_id = 1
t0 = 1 # relative to initial epoch
t1 = 10 # relative to t1
result, man = arp_instance.evaluate_transfer(from_id, to_id, 0, t0, t1)
print (result)

result, man = arp_instance.evaluate_transfer(from_id, to_id, 0, t0, t1, only_cost = True)
print (result)

result, man = arp_instance.evaluate_transfer(from_id, to_id, 0, t0, t1, free_wait = True)
print (result)



# Brute-force
from_id = 1
to_id = 2
t0 = 1
t1 = 10
best = 10e6
best_t1=-1
for t1 in range(5, 730, 1):
    result, man = arp_instance.evaluate_transfer(from_id, to_id, 0, t0, t1)
    if result < best:
        best = result
        best_t1 = t1
        print(f"{t1}:{result}")



# Sequential one dimensional optimization
t0 = 1
from scipy.optimize import minimize_scalar, minimize
best = minimize_scalar(lambda x: arp_instance.evaluate_transfer(from_id, to_id, 0, t0, x)[0],
                       bounds = (1,730), method = 'bounded', options = dict(xatol=1))
print(best)
t1 = int(best.x)
best = minimize_scalar(lambda x: arp_instance.evaluate_transfer(from_id, to_id, 0, x, t1)[0],
                       bounds = (t0,730), method = 'bounded', options = dict(xatol=1))
print(best)

# SLSQP to optimize both.
res = minimize(lambda x: arp_instance.evaluate_transfer(from_id, to_id, 0, x[0], x[1])[0],
               x0 = (0,30),
               bounds = ((0, 730), (1, 730)), method='SLSQP', options=dict(maxiter=50))
print(res)
best, man = arp_instance.evaluate_transfer(from_id, to_id, 0, int(res.x[0]), int(res.x[1]))
print(best)

# Simpler:
solution = arp_instance.optimize_transfer(from_id, to_id, 0, t0_bounds = (0,730), t1_bounds = (1,730))
print(f"t0={solution.ship.leg_times[0]}, t1={solution.ship.leg_times[1]}, f={solution.f}")

solution = arp_instance.optimize_transfer(from_id, to_id, 0, t0_bounds = (0,730), t1_bounds = (1,730), free_wait = True)
print(f"t0={solution.ship.leg_times[0]}, t1={solution.ship.leg_times[1]}, f={solution.f}")

solution = arp_instance.optimize_transfer(-1, 9, 0, t0_bounds = (730,1000), t1_bounds = (1,730))
print(f"t0={solution.ship.leg_times[0]}, t1={solution.ship.leg_times[1]}, f={solution.f}")

solution = arp_instance.optimize_transfer(8, 1, 0, t0_bounds = (0,730), t1_bounds = (0.01,730))
print(f"t0={solution.ship.leg_times[0]}, t1={solution.ship.leg_times[1]}, f={solution.f}")




### Beam search and P-ACO ####################################################
from beam import beam_arp, experiment, rate__orbital_2
from arp_vis import plot_solution

    
# phasing indicator used to rate destination asteroids
rating = rate__orbital_2
    
# reference transfer time (in days) used in the Orbital phasing indicator
ref_dT = 125
    
# "greediness exponent" used in the `heuristic` function
gamma = 50


def go(n, seed, variant, path=None, **kwargs):
    "Launch an experiment"
    
    # define path where experimental results will be saved
    if path is None:
        path = 'results/traj_search/'
    if variant == 'Beam Search':
        path += 'Beam Search '
    
    # configure the number of runs, and stopping criterion
    if variant == 'Beam Search':
        exp_args = dict(
            nr_runs=1, log_data_every=1, max_nr_legs=None, max_nr_gens=1)
    else:
        exp_args = dict(nr_runs=5, log_data_every=2, max_nr_legs=50)
    

    ph = beam_arp(n, seed, rating, ref_dT=ref_dT, gamma=gamma)#,
                         #add_ast_args=add_ast_args)
    exp_args.update(dict(path_handler=ph))
    
    exp_args.update(kwargs)
    
    # RUN experiment
    e = experiment(variant=variant, path=path, **exp_args)
    e.start()
    best = e.return_best()
    return ph, best


# Ordinary beam searches
arp_instance, solution = go(n, seed, 'Beam Search', multiobj=False, beam_width=10, branch_factor=30)
plot_solution(arp_instance, solution.sequence, solution)

arp_instance, solution = go(n, seed, 'Beam Search', multiobj=False, beam_width=20, branch_factor=30)
plot_solution(arp_instance, solution.sequence, solution)


# Stochastic beam searches
arp_instance, solution = go(n, seed, 'Stochastic Beam', multiobj=False, beam_width=10, branch_factor=30)
plot_solution(arp_instance, solution.sequence, solution)

arp_instance, solution = go(n, seed, 'Stochastic Beam', multiobj=False, beam_width=20, branch_factor=30)
plot_solution(arp_instance, solution.sequence, solution)


# Beam P-ACO hybrid searches
arp_instance, solution = go(n, seed, 'Beam P-ACO', multiobj=False, beam_width=20, branch_factor=30, alpha=1, beta=1)
plot_solution(arp_instance, solution.sequence, solution)

arp_instance, solution = go(n, seed, 'Beam P-ACO', multiobj=False, beam_width=20, branch_factor=30, alpha=0.5, beta=1)
plot_solution(arp_instance, solution.sequence, solution)

arp_instance, solution = go(n, seed, 'Beam P-ACO', multiobj=False, beam_width=20, branch_factor=30, alpha=1, beta=0.5)
plot_solution(arp_instance, solution.sequence, solution)


