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

def bracketing(a,s,k,f,g, max_it: float = 10e3):
    """
    Parameters
    ----------
    a : starting point
    s : step size
    k : step size increase
    f : function to be minimized
    g : search direction

    Returns
    -------
    a : one bracket
    c : second bracket
    i : function calls

    """
    state = True
    i = 0
    a_array,b_array,c_array = [],[],[]

    s0,a0 = s, np.copy(a)

    #for dim in range(np.size(a)):
        
    a, b, c = np.copy(a0), np.copy(a0), np.copy(a0)
    
    b[:,:] = b[:,:] + s*g[:,:]
        
    c[:,:] += 2*s*g[:,:]
    
    while state == True:
        # plt.figure(dpi=400)
        # plt.plot(interval, test_fun(interval), 'ko-', linewidth=0.5, markersize = 2)
        # plt.scatter(a, test_fun(a), s=30, c='r', label='A', marker='^')  
        # plt.scatter(b, test_fun(b), s=30, c='b', label='B', marker='^')  
        # plt.scatter(c, test_fun(c), s=30, c='g', label='C', marker='^')  
        # plt.legend()
        # plt.title(f'Iteration {i}')
        
        if (f(a) > f(b)) & (f(c) > f(b)):
            a_array = a[0]
            b_array = b[0]
            c_array = c[0]
            s = s0
            break
        else: 
            a = np.copy(b)
            b = np.copy(c)
            s = s * k
            c[:,:] += s*g[:,:]
        i +=1
        
        if i > max_it:
            state = False
            
    
    return a_array,c_array, i


# test_fun = lambda x: np.sin(x[:,0]) - np.sin(10/3*x[:,0]) + np.sin(x[:,1]) - np.sin(10/3*x[:,1])
# #interval = np.linspace(-1,1)
# a_test, b_test, c_test = bracketing(np.array([[0,0]], dtype=np.float32),0.1,2,test_fun, np.array([[-1,-1]]))
# print(a_test, b_test, c_test)


def sectioning(a, b, f, tol: float = 10e-6, max_steps: int = 10e3):
    
    """
    Parameters
    ----------
    a : intervall1
    b : intervall2
    f : function to be minimized
    tol : tolerance, stopping criterion
    max_steps: max number of steps, stopping criterion

    Returns
    -------
    a_new : left side
    b_new : right side
    counter : function calls
    
    """
    
    
    value1, value2 = [],[]
    
    
    # unit_vec = np.array([[0]*np.size(a)])
    # unit_vec[:,dim] = 1
    a_new, b_new = np.array([a]), np.array([b])

    c = b_new - rho*(b_new-a_new)
    d = a_new + rho*(b_new-a_new)
    
    running = True
    counter = 0
    
    while running:
        if f(c) < f(d):
            b_new = d
            d = np.copy(c)
            c = b_new - rho*(b_new-a_new)
        else:
            a_new = np.copy(c)
            c = d
            d = a_new + rho*(b_new-a_new)
        
        if np.all(abs(a_new-b_new) < tol):
            running = False
            print("Threshold reached")

            
        if counter > max_steps:
            running = False
            print("Max. step reached")
                    
        counter += 1
            
    return a_new,b_new,counter


#sol = sectioning(a_test, c_test, test_fun)
# test_sol = sectioning(a_test, c_test,test_fun, tol = 10e-5)
# print(test_sol)

testfun = test_function(1)[0]
intervall1,intervall2, n1 = bracketing(np.array([[10,-10]], dtype = np.float32),2,1.01,testfun,np.array([[-1,1]], dtype = np.float32))
a_new, b_new, n2 = sectioning(intervall1, intervall2, testfun)
left_side = testfun(a_new*0.90)[0]
middle = testfun(a_new)[0]
right_side = testfun(a_new*1.1)[0]
print(f"left-side: {left_side:.4f}, value: {middle:.4f}, right side: {right_side:.4f}")
print(f"check square: x_1 = {a_new[0,0]**2:.4f}, x2 =  {a_new[0,1]:.4f}")


############ CONTROL FUNCTIONS ################ 
# Please do not change these functions as the #
# grading will not be possible.               #
###############################################

def get_test_fun_list() -> List[Callable]:
    return [line_search]
