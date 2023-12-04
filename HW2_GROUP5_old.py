from typing import Callable, List, Tuple
import numpy as np
import test_functions as tf




def print_available():
    print("Available functions with gradients and hessians : \n")
    print("  - rosenbrock (id=1) ndim = 2")
    print("  - himmelblau (id=2) ndim = 2")
    print("  - rastrigin (id=3) ndim = 2")
    print("  - F6 (id=4) ndim = 1")
    print("  - threehumpcamel (id=5) ndim = 2")

def cx(x):

    if len(x.shape) == 1:
        x = x[:,None]
    elif x.shape[0] != 2:
        x = x.T

    return x

############ Previously implemented functions #################
############ Global variables #################
rho: float = 0.61803398875
state: str = True
############ Helper functions #################
def CalculateJacobian(model, x0, h):
    model_x0 = model(x0)
    dim = (np.shape(model_x0)[0], np.shape(x0)[0])
    J = np.zeros(dim)
    for xi in range(len(x0)):
        x0_der = x0.copy()
        x0_der[xi] = x0_der[xi] + h
        J[:,xi] = (model(x0_der) - model_x0)[:,0]/h
    return J
    
    


def bracketing(
        a: float,
        s: float,
        k: float,
        f: Callable,
        max_it: float = 10e3,
        ):
    b = a + s
    i = 1

    if f(a)>f(b):
        pass
    else:
        a,b = b,a
        s = -s
    c = b + s

    while state == True:
        i +=1
        if f(c) > f(b):
            break
        elif i > max_it:
            print("Max. iterations reached!")
            break
        else:
            a = b
            b = c
            s = s * k
            c = b + s
    return a,b,c,i


def sectioning(
        a: float,
        b: float,
        c: float,
        f: Callable,
        tol: float = 1e-10,
        max_it: float = 10e3,
        ):

    c_range = c - rho*(c-a)
    a_range = a + rho*(c-a)

    counter = 1

    while state == True:

        if f(c_range) < f(a_range):
            c = a_range
            a_range = c_range
            c_range = c - rho*(c-a)
        elif counter > max_it:
            print("Max. iterations reached!")
            break    
        else:
            a = c_range
            c_range = a_range
            a_range = a + rho*(c-a)

        if abs(a-c) < tol:
            print("\nThreshold reached")
            break
        counter += 1
    return a,c,counter
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

    a, b, c, i_run = bracketing(alpha0,s,k,line_fun)
    a,c, counter = sectioning(a, b, c,line_fun)

    f_calls = i_run + counter


    return a,c, f_calls
############ CO


############ Solution functions ###############
############ FIRST ORDER METHODS ##############


def gradient_descent(
    f: Callable[[np.ndarray], np.ndarray],
    g: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    s: float = 2e-4,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Implementation of gradient descent algorithm.

    Args:
        f (Callable): Objective function
        g (Callable): Gradient of objective function
        x0 (np.ndarray): Starting point
        s (float, optional): Line search start step size. Defaults to 0.01.
        eps (float): Optimalty condition parameter.

    Returns:
        np.ndarray: optimal point.
        np.ndarray: list of steps the method takes to the optimal point (last one must be the optimal point from return 1).
        int: number of forward function calls.
        int: number of gradient evaluations. This is also the number of iterations + 1.
    """

    ##### Write your code here (Values below just as placeholders if the 
    # function in question is not implemented by you for some reason, make sure
    # that you leave such a definition since the grading wont work. ) #####
    #x_steps = [x0]
    #x_opt = x0.copy()
    #fc = gc = 0
    
    x_steps = []
    fc = gc = 0 
    
    x_opt = np.copy(x0)
    counter = 0
    
    while np.any(np.abs(g(x_opt)) > eps) and counter < 10000:
        print(g(x_opt))
        #print(f(x_opt))
        #print(x_opt)
        d = -g(x_opt)
        a,b,run = line_search(f, alpha0 = 0, x0 = x0, g = d, s = s, eps = eps)
        print('a =', a, 'b = ', b)
        alpha = (a+b)/2
        x_opt = x_opt + alpha*d
        x_steps.append(x_opt.copy())
        gc += 1 
        fc += run
        counter += 1
        print(counter)
    
    return x_opt, x_steps, fc, gc


def gauss_newton(
    f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-5,
    h: float = 1e-8,
    N_max: int = 1_000,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Implementation of the Gauss-Newton method.

    Args:
        f (Callable): Objective function.
        x0 (np.ndarray): Initial objective function parameters.
        y (np.ndarray): Goal values for fitting.
        eps (float): Optimalty condition parameter. Defaults to 1e-5.
        h (float): finite difference step size. Defaults to 1e-8.
        N_max (int): maximal number of iterations. Defaults to 1000.

    Returns:
        np.ndarray: optimal point.
        np.ndarray: list of steps the method takes to the optimal point (last one must be the optimal point from return 1).
        int: number of forward function calls.
    """

    
    
    ##### Write your code here (Values below just as placeholders if the 
    # function in question is not implemented by you for some reason, make sure
    # that you leave such a definition since the grading wont work. ) #####
    x_steps = [x0]
    x_opt = x0.copy()
    n_iter = 0
    state = True
    
    while state == True:
        J = CalculateJacobian(f, x_opt, h)
        J_T = J.T
        r = - y + f(x_opt)
        x_opt = x_opt - np.linalg.inv(J_T@J)@J_T@r
        
        if n_iter > N_max:
            state = False
        if np.all(abs(x_opt - x_steps[n_iter]))<eps:
            state = False    
        
        x_steps.append(x_opt)
        n_iter += 1 
        
        
    return x_opt, x_steps, n_iter


############ SECOND ORDER METHODS #############


def newton(
    g: Callable[[np.ndarray], np.ndarray],
    G: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    eps=1e-5,
    N_max=1000,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Implementation of the Newton method.

    Args:
        g (Callable): Gradient of objective function.
        G (Callable): Hessian of objective function.
        x0 (np.ndarray): Starting point.
        eps (float): Optimalty condition parameter. Defaults to 1e-5.
        N_max (int): maximal number of iterations. Defaults to 1000.

    Returns:
        np.ndarray: optimal point.
        np.ndarray: list of steps the method takes to the optimal point (last one must be the optimal point from return 1).
        int: number of forward function calls.
    """

    ##### Write your code here (Values below just as placeholders if the 
    # function in question is not implemented by you for some reason, make sure
    # that you leave such a definition since the grading wont work. ) #####
    # x_steps = [x0]
    # x_opt = x0.copy()
    # n_iter = 0
    
    x_steps = []
    n_iter = 0
    
    condition = True
    x_old = x_opt = x0.copy()

    
    while condition is True:
        G_xold = G(x_old)
        x_opt = x_old - np.linalg.inv(G_xold)@g(x_old)
        
        if n_iter > N_max:
            condition = False
            print('Max iteration reached')
        if np.all(np.abs(x_opt-x_old) < eps):
            if all(np.linalg.eigvals(G_xold)>0):
                condition = False
                print('Convergence limit reached')
            
        x_steps.append(x_opt) 
        x_old = x_opt.copy()
        n_iter += 1

    return x_opt, x_steps, n_iter


def levenberg_marquardt(
    g: Callable[[np.ndarray], np.ndarray],
    G: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    nu: float = 1e-3,
    eps: float = 1e-2,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Implementation of the Levenberg-Marquardt method.

    Args:
        g (Callable): Gradient of objective function.
        G (Callable): Hessian of objective function.
        x0 (np.ndarray): Starting point.
        nu (float): Regularization parameter.
        eps (float): Optimalty condition parameter.

    Returns:
        np.ndarray: optimal point.
        np.ndarray: list of steps the method takes to the optimal point (last one must be the optimal point from return 1).
        int: number of forward function calls.
    """

    ##### Write your code here (Values below just as placeholders if the 
    # function in question is not implemented by you for some reason, make sure
    # that you leave such a definition since the grading wont work. ) #####
    # x_steps = [x0]
    # x_opt = x0.copy()
    # n_iter = 0
    G_x0 = G(x0)
    # D = D_inv = np.zeros(G_x0.shape)
    # for i in range(G_x0.shape[1]):
    #     d_i =  np.sqrt(G_x0[i,:].sum())
    #     D[i,i] = d_i
    #     D_inv[i,i] = 1/d_i
    # X_0 = D@G_x0@D + nu*np.identity(G_x0.shape[1])
    # b_0 =   D_inv@g(x0)
    
    #p_0 = -D@(np.linalg.inv(X_0)@b_0)
    G_new = lambda x: G(x) + nu*np.identity(G_x0.shape[1])
    p_0 = np.linalg.inv(G_x0+nu*np.identity(G_x0.shape[1]))@x0
    x1 = x0 + p_0
    
    x_opt, x_steps, n_iter = newton(g,G_new,x1)
    

    return x_opt, x_steps, n_iter


f,g,_,_,_,_,_ = tf(1)

x_test, test_steps, fc,gc = gradient_descent(f,g,np.array([[1],[1]]))


testgrad = lambda x: np.array([3*x[0]**2, 2*x[1]])
testhessian = lambda x: np.array([[6*x[0],0],[0,2]])
x_opt, x_steps, nit = newton(testgrad, testhessian, np.array([-15,-5], dtype = np.float64))

testgrad = lambda x: np.array([4*x[0]**3+x[1], x[0]+2*(1+x[1])])
testhess = lambda x: np.array([[4*x[0]**2, 1], [1,2]])

x_opt, x_steps, nit = levenberg_marquardt(testgrad, testhess, np.array([15,5], dtype = np.float64), nu = 0)

## TEST OF NEWTON FUNCTION
# f,g,H = test_function(1)[:3]
# x0 = np.array([[0,-1]]).T
# xs,xstep,n_iter = newton(g,H,x0)
# print('optimum at x =[', xs[0,0],',', xs[1,0], ']')


# ## TEST OF LEVENBERG FUNCTION
# f,g,H = test_function(2)[:3]
# x0 = np.array([[0.75,-1.25]]).T
# xs,xstep,n_iter = levenberg_marquardt(g,H,x0, nu = 50)
# print('optimum at x =[', xs[0,0],',', xs[1,0], ']')

## TEST OF GRADIENT DESCENT'
# f,g = test_function(1)[:2]
# #f = lambda x: np.sin(x) - np.sin(10/3*x)
# #g = lambda x: np.cos(x) - np.cos(10/3*x)*10/3

# #f = lambda x: x[0,:]**2 + x[1,:]**2
# #g = lambda x: np.array([2*x[0,:]**1 , 2*x[1,:]**1])
# x0 = np.array([[3,-2]], dtype = np.float64).T
# x_opt, x_steps, fc, gc = gradient_descent(f, g, x0, s = 1e-10)
# a,b,run = line_search(f, alpha0 = 0, x0 = x0, g = g(x0))


# ## TESTING GAUSS NEWTON METHODE
# import scipy.io as sio
# import matplotlib.pyplot as plt
# I = sio.loadmat(r"C:\Users\bened\Desktop\Numerical Optimization\HW2_files\HW-2\I_m.mat")["I_m"].T
# f = np.logspace(1, 6, len(I))[:,None]
# omega = 2*np.pi*f
# model = lambda x: np.abs(1/(x[0]+ 1j*(omega*x[1]-1/(omega*x[2]))))
# x0 = np.array([[0.5],[2e-3],[3e-6]]) # scaled values, R=x(0), L=x(1).10^-3, C=x(2).10^-
# xs, xsteps, fc = gauss_newton(model, x0.copy(), np.abs(I), N_max=1_000, h = 1e-8)

# fig, ax = plt.subplots(figsize=((9,7)))
# ax.scatter(f, abs(I),label='measurements, y')
# ax.set_xscale('log'); plt.xlabel('f');plt.ylabel('abs(I)'); plt.title('Resonance curve of a series reson')
# ax.plot(f, model(xs),color="grey",label='model, F(x)')
# ax.legend()

########### CONTROL FUNCTIONS ################


def get_test_fun_list() -> List[Callable]:
    return [gradient_descent, gauss_newton, newton, levenberg_marquardt]
