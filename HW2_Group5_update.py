from typing import Callable, List, Tuple
import numpy as np
from numpy.linalg import norm


############ Global variables #################
rho: float = 0.61803398875
state: str = True
############ Helper functions #################

############ Previously implemented functions #################
def bracketing(
        a: float,
        s: float,
        k: float,
        f: Callable,
        ):
    b = a + s
    i = 0

    fa = f(a)
    fb = f(b)
    i += 2

    if fa<fb:
        fa, fb = fb, fa
        a,b = b,a
        s = -s
    else:
        pass

    c = b + s

    fc = f(c)
    i +=1
    while state == True:
        if fc > fb:
            i +=1
            break
        else:
            a = b
            b = c
            s = s * k
            c = b + s

            fb = fc
            fc = f(c)

    return a,b,c,i

def sectioning(
        a: float,
        b: float,
        c: float,
        f: Callable,
        eps: float,
        ):

    c_range = c - rho*(c-a)
    a_range = a + rho*(c-a)

    counter = 0
    fc_range = f(c_range)
    fa_range = f(a_range)

    counter +=2

    while state == True:

        if fc_range < fa_range:
            c = a_range
            a_range = c_range
            c_range = c - rho*(c-a)


        else:
            a = c_range
            c_range = a_range
            a_range = a + rho*(c-a)

        if abs(a-c) < eps:
            break


    return a,c,counter


# Line search function :
def line_search(
    f: Callable,
    alpha0: float,
    x0: np.ndarray,
    g: np.ndarray,
    s: float = 2e-4,
    k: float = 3.0,
    eps: float = 1e-3,
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
    a,c, counter = sectioning(a, b, c,line_fun, eps)

    f_calls = i_run + counter


    return a,c, f_calls

############ Solution functions ###############
############ FIRST ORDER METHODS ##############
def gradient_descent(
    f: Callable[[np.ndarray], np.ndarray],
    g: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    s: float = 2e-4,
    eps: float = 1e-3,
    k: float = 3,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Implementation of gradient descent algorithm.

    Args:
        f (Callable): Objective function
        g (Callable): Gradient of objective function
        x0 (np.ndarray): Starting point
        s (float, optional): Line search start step size. Defaults to 0.01.
        eps (float): Optimalty condition parameter.
        k: line search mulitplier

    Returns:
        np.ndarray: optimal point.
        np.ndarray: list of steps the method takes to the optimal point (last one must be the optimal point from return 1).
        int: number of forward function calls.
        int: number of gradient evaluations. This is also the number of iterations + 1.
    """
    x_steps = [x0]
    x_opt = x0.copy()
    fc = 0
    gc = 1

    while norm(g(x_opt)) >= eps:
        dn = -g(x_opt).T
        a,b, f_calls = line_search(alpha0=0, x0=x_opt, g=dn ,f = f, s = s, k = k)

        # Take the mean value of the bounds as the optimal distance
        alpha = 0.5*(a + b)

        # Minimizer x*
        x_opt = x_opt + alpha*dn

        x_steps.append(x_opt)

        fc = fc + f_calls
        gc += 1

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

    return x_opt, x_steps, n_iter


############ SECOND ORDER METHODS #############
def newton(
    g: Callable[[np.ndarray], np.ndarray],
    G: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    eps=1e-3,
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
    x_steps = [x0]
    x_opt = x0.copy()
    n_iter = 1
    state_imp = True

    while state_imp:
        n_iter += 1
        p = np.dot(np.linalg.inv(G(x_opt)), g(x_opt)).reshape(1, 2)
        x_opt = x_opt - p
        x_steps.append(x_opt)

        if n_iter == N_max:
            print('Max. iterations reached!')
            break

        elif np.linalg.norm(g(x_opt)) < eps and np.all(np.linalg.eigvals(G(x0))):
            print("Convergence reached!")
            break


    return x_opt, x_steps, n_iter


def levenberg_marquardt(
    g: Callable[[np.ndarray], np.ndarray],
    G: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    nu: float = 1e-3,
    eps: float = 1e-3,
    N_max=1000,
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

    x_steps = [x0]
    x_opt = x0.copy()
    n_iter = 1
    nu_default = nu

    while np.linalg.norm(g(x_opt)) >= eps:
        Hessian = G(x_opt)
        eigvals = np.linalg.eigvals(Hessian)

        nu = nu_default

        while np.all(eigvals < 0):
            Hessian = Hessian + nu * np.eye(len(Hessian))
            eigvals = np.linalg.eigvals(Hessian)
            nu *= 2

        n_iter += 1
        p = np.dot(np.linalg.inv(Hessian), g(x_opt)).reshape(1, 2)
        x_opt = x_opt - p
        x_steps.append(x_opt)

        if n_iter == N_max:
            print('Max. iterations reached!')
            break

    print("Convergence reached!")


    return x_opt, x_steps, n_iter



# import scipy.io as sio
# I = sio.loadmat("I_m.mat")["I_m"].T
# f = np.logspace(1, 6, len(I))[:,None]

# x0 = np.array([[0.5],[0.2],[0.3]]) # scaled values, R=x(0), L=x(1).10^-3, C=x(2).10^-6
# # xs, xsteps, fc = gauss_newton(model_eq, x0.copy(), np.abs(I),  N_max=1_000)[:3]
# #print(xs)

# fig, ax = plt.subplots(figsize=((9,7)))
# ax.scatter(f, abs(I),label='measurements, y')
# ax.set_xscale('log'); plt.xlabel('f');plt.ylabel('abs(I)'); plt.title('Resonance curve of a series resonant circuit')
# # ax.plot(f, model_eq(xsteps[:][3]),color="grey",label='model, F(x)')
# ax.legend()


def get_test_fun_list() -> List[Callable]:
    return [gradient_descent, gauss_newton, newton, levenberg_marquardt]
