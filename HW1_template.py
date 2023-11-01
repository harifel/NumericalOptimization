from typing import Callable, List, Tuple
from dataclasses import dataclass
import numpy as np

############ Solution functions ################
# The function bodies in the following section #
# are to be edited. The function input params  #
# and return values along with the function    #
# name must not change as well as the imports  #
################################################

############ Global variables #################

rho: float = 0.61803398875

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
    state = True
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
        plt.scatter(a, test_fun(a), s=30, c='r', label='A', marker='^')  
        plt.scatter(b, test_fun(b), s=30, c='b', label='B', marker='^')  
        plt.scatter(c, test_fun(c), s=30, c='g', label='C', marker='^')  
        plt.legend()
        plt.title(f'Iteration {i}')
        i +=1
        if f(c) > f(b):
            break
        else: 
            a = b
            b = c
            s = s * k
            c = b + s
    return a,b,c


test_fun = lambda x: np.sin(x) - np.sin(10/3*x)
interval = np.linspace(-1,1)
a_test, b_test, c_test = bracketing(0,0.1,2,test_fun)
print(a_test, b_test, c_test)


def sectioning(a, b, f, tol: float = 10e-6, max_steps: int = 10e3):
    a_new = a
    b_new = b
    c = b - rho*(b-a)
    d = a + rho*(b-a)
    
    running = True
    counter = 0
    
    while running:
        if f(c) < f(d):
            b_new = d
            d = c
            c = b_new - rho*(b_new-a_new)
        else:
            a_new = c
            c = d
            d = a_new + rho*(b_new-a_new)
        
        if abs(a_new-b_new) < tol:
            running = False
            print("Threshold reached")
            
        if counter > max_steps:
            running = False
            print("Max. step reached")
                    
        counter += 1
            
    return (a_new+b_new)/2

# test_sol = sectioning(a_test, c_test,test_fun, tol = 10e-5)
# print(test_sol)

############ CONTROL FUNCTIONS ################ 
# Please do not change these functions as the #
# grading will not be possible.               #
###############################################

def get_test_fun_list() -> List[Callable]:
    return [line_search]
