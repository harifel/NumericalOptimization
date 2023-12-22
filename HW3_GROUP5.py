from typing import Callable, List, Tuple
import numpy as np
import test_functions as tf 

import numpy as np
import scipy.io as sio
import pybisol
from helpers import plot_geometry

############ Global variables #################

###############################################
############ Helper functions #################
def normalize(x, norm):
    x_norm = (x - norm[0,:]) / (norm[1,:] - norm[0,:])
    return x_norm

def de_normalize(x, norm):
    x_norm = x * (norm[1,:] - norm[0,:]) + norm[0,:] 
    return x_norm
###############################################

def f_penalized(f: Callable, p_list: List[Callable], x: np.ndarray, rho: float=2.0) -> float:
    """ F Penalized function implementing the penalty constraints method.

    Args:
        f (Callable): objective function.
        p_list (List[Callable]): list containing penalty functions.
        x (np.ndarray): point where to evaluate the functinos. 
        rho (float, optional): Penalty starting value. Defaults to 2.0.

    Returns:
        [float]: Value of the penalized objective.
    """    
    
    # Evaluate the objective function
    f_value = f(x)
    
    # Add penalty terms based on constraint violations
    penalty = sum(max(0, p(x))**2 for p in p_list)
    
    # Calculate the penalized objective
    f_p = f_value + rho * penalty

    return f_p

def pattern_search(f: Callable, x0: np.ndarray, ranges: np.ndarray, alpha: float=0.1, gamma: float=0.5, delta: float=0.1, eps: float=1e-3, max_iter: int=10_000, verbose:bool=False, N_print: int=10):
    """ Implementation of the pattern search function.

    Args:
        f (Callable): objective function.
        x0 (np.ndarray): starting point with shape 1xN.
        ranges (np.ndarray): range of design variables 2xN.
        alpha (float, optional): Exploration step size. Defaults to 0.1.
        gamma (float, optional): Step size reduction factor. Defaults to 0.5.
        delta (float, optional): Extrapolation step size. Defaults to 0.1.
        eps (float, optional): Termination value. Defaults to 1e-2.
        max_iter (int, optional): Maximal numer of iterations. Defaults to 1_000.
        verbose (bool, optional): Printing iteration info. Defaults to False.

    Returns:
        [Tuple[np.ndarray, List]]: [optimal point, optimization steps]
    """    
    
    x_steps = [x0]
    x0 = normalize(x0, ranges)
    x_opt = x0.copy()
    
    n_iter = 0
    state = True
    
    
    
    while state:
        #exploration

        #x_opt = x_trial
        x_trial = x_opt.copy()
        
        f0 = f(de_normalize(x_trial, ranges))
        
        # exploration = True
        
        x1 = []
        f_x1 = []
        # x2 = []
        # f_x2 = []
        
        # while exploration:
        #     x1_plus = x_trial + alpha * np.array([1, 0])
        #     x1_minus = x_trial - alpha * np.array([1, 0])
            
        #     x1.append(x1_plus)
        #     x1.append(x1_minus)
            
        #     f_x1.append(f(de_normalize(x1_plus, ranges)))
        #     f_x1.append(f(de_normalize(x1_minus, ranges)))

        #     if f0 > f_x1[0] or f0 > f_x1[1]:
        #         min_index = np.argmin(np.abs(f_x1))
        #         x_trial = x1[min_index]
                       
        #     x2_plus = x_trial + alpha * np.array([0, 1])
        #     x2_minus = x_trial - alpha * np.array([0, 1])
            
        #     x2.append(x2_plus)
        #     x2.append(x2_minus)
            
        #     f_x2.append(f(de_normalize(x2_plus, ranges)))
        #     f_x2.append(f(de_normalize(x2_minus, ranges)))
        
        #     if f(de_normalize(x_trial, ranges)) > f_x2[0] or f(de_normalize(x_trial, ranges)) > f_x2[1]:
        #         min_index = np.argmin(f_x2)
        #         x_trial = x2[min_index]
            
        #     exploration = False
        #     i = 0
        
        i = 0
        x_trial = x_opt
        while i < len(x_opt[0]):
            x1 = []
            f_x1 = []
            
            f0 = f(de_normalize(x_trial, ranges))
                        
            x_inter = np.zeros(x_trial.shape)
            x_inter[:, i] = 1
            x1_plus = x_trial + alpha * x_inter
            x1_minus = x_trial - alpha * x_inter

            x1.append(x1_plus)
            x1.append(x1_minus)
            
            f_x1.append(f(de_normalize(x1_plus, ranges)))
            f_x1.append(f(de_normalize(x1_minus, ranges)))
            
            if f0 > f_x1[0] or f0 > f_x1[1]:
                min_index = np.argmin(np.abs(f_x1))
                x_trial = x1[min_index]
            
            i = i + 1
            
                

        #extrapolation
        if f(de_normalize(x_trial,ranges)) < f0:    
            #x_trial = x_trial + delta * (x_trial - x_opt)
            x_opt = x_trial + delta * (x_trial - x_opt)
            x_steps.append(de_normalize(x_trial, ranges))
            
        else:
            alpha *= gamma
            delta *= gamma
            
        if verbose == True and n_iter % N_print == 0:
            print(f'Iter: {n_iter} | y_opt = [{f(de_normalize(x_opt, ranges))}] | alpha = {alpha} | delta = {delta}')    
            
        n_iter += 1
        if n_iter == max_iter:
            break
            
        if alpha < eps and delta < eps:
            break
        
        
    return de_normalize(x_opt, ranges), x_steps


############ CONTROL FUNCTIONS ################

def get_test_fun_list() -> List[Callable]:
    return [pattern_search, f_penalized]

###############################################
