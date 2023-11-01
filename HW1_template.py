from typing import Callable, List, Tuple
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

############ Solution functions ################
# The function bodies in the following section #
# are to be edited. The function input params  #
# and return values along with the function    #
# name must not change as well as the imports  #
################################################

############ Global variables #################
rho: float = 0.61803398875
state: str = True
############ Helper functions #################
# Can be implemented here if needed.          #
###############################################

########### Main solution function : ##########

def line_search(
    f: Callable,
    alpha0: float,
    x0: np.ndarray,
    g: np.ndarray,
    s: float = 10e-2,
    k: float = 2.0,
    eps: float = 1e-2,
) -> Tuple[float, float, int]:
    """[summary]

    Args:
        f (Callable): Function to perform the line search on.
        alpha0 (float): Initial step parameter.
        x0 (np.ndarray): Starting position.
        g (np.ndarray): Search direction.
        s (float, optional): Line search step scalar. Defaults to 10e-2.
        k (float, optional): Line search step expansion. Defaults to 2.0.
        eps (float, optional): Termination condition eps. Defaults to 1e-2.

    Returns:
        Tuple[float, float, int]: bracket left, bracket right, number of function calls
    """    
    line_fun = lambda alpha: f(x0 + alpha*g)
    
    return a, b, f_calls

def bracketing(a,s,k,f):
    """
    Parameters
    ----------
    a : starting point
    s : step size
    k : step size increase
    f : function to be minimized

    Returns
    -------
    a : one bracket
    c : second bracket

    """
    
    i = 0

    b = a + s
    
    #define search direction g
    if f(a)>f(b):
        pass
    else:
        a,b = b,a
        s = -s
    c = b + s
    
    while state == True:
        plt.figure(dpi=400)
        plt.plot(interval, test_fun(interval), 'ko-', linewidth=0.5, markersize = 2)
        plt.scatter(a, f(a), s=30, c='r', label='A', marker='^')  
        plt.scatter(b, f(b), s=30, c='b', label='B', marker='^')  
        plt.scatter(c, f(c), s=30, c='g', label='C', marker='^')  
        plt.legend()
        plt.title(f'Bracketing - Iteration {i}')
        i +=1
        if f(c) > f(b):
            break
        else: 
            a = b
            b = c
            s = s * k
            c = b + s
    return a,b,c,i


test_fun = lambda x: np.sin(x) - np.sin(10/3*x)
interval = np.linspace(-3,1)
a_test, b_test, c_test, i_run = bracketing(0,1,2,test_fun)

print(f"a,b,c-values: {a_test:.4f}, {b_test:.4f}, {c_test:.4f}")
print(f"Function calls:: {(i_run):.4f}")


def sectioning(a, b, c ,f, tol: float = 10e-3):

    c_range = c - rho*(c-a)
    a_range = a + rho*(c-a)
    
    counter = 0

    while state == True:
        
        plt.figure(dpi=400)
        plt.plot(interval, test_fun(interval), 'ko-', linewidth=0.5, markersize = 2)
        plt.scatter(a, f(a), s=30, c='r', label='A', marker='^')  
        plt.scatter(c_range, f(c_range), s=60, c='b', label='B (c_range)', marker='*')  
        plt.scatter(a_range, f(a_range), s=60, c='b', label='B (a_range)', marker='+')  
        plt.scatter(c, f(c), s=30, c='g', label='C', marker='^')  
        plt.legend()
        plt.title(f'Bracketing - Iteration {i_run-1}, Sectioning - Counter {counter}')
        
        if f(c_range) < f(a_range):
            c = a_range
            a_range = c_range

            c_range = c - rho*(c-a)
        
        else:
            a = c_range
            c_range = a_range
            
            a_range = a + rho*(c-a)
        
        if abs(a-c) < tol:
            print("\nThreshold reached")
            break
             
        counter += 1
    
    return (a-c), (a+c)/2, counter


error, test_point, counter = sectioning(a_test, b_test, c_test,test_fun)
print(f"Error: {np.abs(error):.4f}")
print(f"Minimum x-value:: {test_point:.4f}")
print(f"Function calls:: {(counter):.4f}")




############ CONTROL FUNCTIONS ################ 
# Please do not change these functions as the #
# grading will not be possible.               #
###############################################

def get_test_fun_list() -> List[Callable]:
    return [line_search]