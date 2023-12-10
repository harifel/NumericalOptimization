from typing import Callable, List, Tuple
import numpy as np
import test_functions as tf 

############ Global variables #################

###############################################
############ Helper functions #################

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
    #f_p = 0
    
    # Evaluate the objective function
    f_value = f(x)
    
    # Add penalty terms based on constraint violations
    penalty = sum(max(0, p(x)) for p in p_list)
    
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
    #x_opt = x_steps = []
    
    def Exploration(f, x_opt, alpha):
        x_exp = np.zeros(np.shape(x_opt))
        x_prop = x_opt.copy()
        f0 = f(x_opt)
        for n in range(max(np.shape(x_opt))):

            x_prop[:,n] = x_opt[:,n] + alpha#*abs(ranges[1,n]- ranges[0,n])
            f1 = f(x_prop)
            if f1 < f0:
                x_exp[:,n] = x_opt[:,n] + alpha#*abs(ranges[1,n]- ranges[0,n])
                f0 = f1.copy()
            else:
                x_prop[:,n] = x_opt[:,n] - alpha#*abs(ranges[1,n]- ranges[0,n])
                f2 = f(x_prop)
                if f2 < f0:
                    x_exp[:,n] = x_opt[:,n] - alpha#*abs(ranges[1,n]- ranges[0,n])
                    f0 = f2.copy()
                else:
                    x_exp[:,n] = x_opt[:,n]
        return x_exp
    
    
    
    x_opt = x0.copy()
    x_steps = [x_opt]
    n_iter = 0
    state = True
    x_exp_new = Exploration(f, x_opt, alpha)
    f0 = f(x_exp_new)
    x_exp = x_opt.copy()
    
    while state == True:
        
        if verbose == True:
            if n_iter % N_print == 0:
                print(f'Iter: {n_iter} | y_opt = [{f(x_opt)}] | alpha = {alpha} | delta = {delta}')
        
        x_opt_new = x_exp + delta*(x_exp_new-x_exp)
        x_exp = Exploration(f, x_opt_new, alpha)
        f1 = f(x_exp)
        if f1 < f0:
            x_opt = x_opt_new.copy()
            x_steps.append(x_opt)
            f0 = f1.copy()
            x_exp, x_exp_new = x_exp_new.copy(), x_exp.copy()
#            print(x_opt)
        else:
            alpha *= gamma
            delta *= gamma
            x_steps.append(x_opt_new)
        

        
        n_iter += 1 
        if n_iter > max_iter:
            state = False
        
        if alpha < eps:
            state = False
            
        
    return x_opt, x_steps


### TEST PATTERN SEARCH

x0 = np.array([[-4,-4]], dtype=np.float32)
ranges = np.array([[-5,-5],[5, 5]], dtype=np.float32)
obj = lambda x: tf.test_function(1)[0](x.T)
x_opt, x_steps = pattern_search(obj, x0, ranges, alpha=.1, eps=1e-6, verbose=True, N_print=100, delta = .1, max_iter = 80000)
print(x_opt)

## Test penalty method



# obj = lambda x: tf.test_function(2)[0](x.T)

############ CONTROL FUNCTIONS ################

def get_test_fun_list() -> List[Callable]:
    return [pattern_search, f_penalized]

###############################################
