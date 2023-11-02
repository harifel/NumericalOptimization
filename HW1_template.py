from typing import Callable
import numpy as np


def print_available():
    print("Available functions with gradients and hessians : \n")
    print("  - rosenbrock (id=1) ndim = 2")
    print("  - himmelblau (id=2) ndim = 2")
    print("  - rastrigin (id=3) ndim = 2")
    print("  - F6 (id=4) ndim = 1")
    print("  - threehumpcamel (id=5) ndim = 2")


def test_function(fun_id):

    f: Callable = None
    g: Callable = None
    H: Callable = None
    n_x: int = None
    limits: np.ndarray = None
    obj: np.ndarray = None
    x_opt: np.ndarray = None

    if fun_id == "rosenbrock" or fun_id == 1:
        #  Rosenbrock's function
        #   Minimum: f(1,1) = 0
        f = lambda x: (1 - x[:, 0]) ** 2 + 100 * (x[:, 1] - x[:, 0] ** 2) ** 2
        g = lambda x: np.array(
            [
                -400 * x[:, 0] * (x[:, 1] - x[:, 0] ** 2) - 2 * (1 - x[:, 0]),
                200 * (x[:, 1] - x[:, 0] ** 2)
            ]
        )
        H = lambda x: np.array(
            [
                (1200 * x[:, 0] ** 2 - 400 * x[:, 1] + 2)[0],
                (-400 * x[:, 0])[0],
                (-400 * x[:, 0])[0],
                200,
            ]
        ).reshape(2, 2)
        n_x = 2
        limits = np.array([[-5, 5], [-5, 5]], dtype=float)
        obj = 0
        x_opt = np.array([[1, 1]])

    elif fun_id == "himmelblau" or fun_id == 2:
        #  Himmelblau's function
        #   Minimum:  * f( 3,         2       ) = 0
        #             * f(-2.805118,  3.131312) = 0
        #             * f(-3.779310, -3.283186) = 0
        #             * f( 3.584428, -1.848126) = 0
        f = (
            lambda x: (x[:, 0] ** 2 + x[:, 1] - 11) ** 2
            + (x[:, 0] + x[:, 1] ** 2 - 7) ** 2
        )

        g = lambda x: np.array(
            [
                4 * x[:, 0] * (x[:, 0] ** 2 + x[:, 1] - 11)
                + 2 * (x[:, 0] + x[:, 1] ** 2 - 7),
                2 * (x[:, 0] ** 2 + x[:, 1] - 11)
                + 4 * x[:, 1] * (x[:, 0] + x[:, 1] ** 2 - 7),
            ]
        )

        H = lambda x: np.array(
            [
                [
                    (12 * x[:, 0] ** 2 + 4 * x[:, 1] - 42)[0],
                    (4 * x[:, 0] + 4 * x[:, 1])[0],
                ],
                [
                    (4 * x[:, 0] + 4 * x[:, 1])[0],
                    (4 * x[:, 0] + 12 * x[:, 1] ** 2 - 26)[0],
                ],
            ]
        )
        n_x = 2
        # 'n_x' states
        limits = np.array([[-5, 5], [-5, 5]], dtype=float)  # Boundaries
        obj = np.zeros((4,))
        # objective value (f(x_min) = obj)
        x_opt = np.array(
            [
                [3, 2],
                [-2.805118, 3.131312],
                [-3.779310, -3.283186],
                [3.584428, -1.848126],
            ]
        )
    elif fun_id == "rastrigin" or fun_id == 3:
        #  Rastrigin function
        #   Minimum:  f(0,0) = 0
        f = (
            lambda x: 20
            + (x[:, 0] ** 2 + x[:, 1] ** 2)
            - 10 * (np.cos(2 * np.pi * x[:, 0]) + np.cos(2 * np.pi * x[:, 1]))
        )

        g = lambda x: np.array(
            [
                2 * x[:, 0] + 20 * np.pi * np.sin(2 * np.pi * x[:, 0]),
                2 * x[:, 1] + 20 * np.pi * np.sin(2 * np.pi * x[:, 1]),
            ]
        )

        H = lambda x: np.array(
            [
                [(2 + 40 * np.pi ** 2 * np.cos(2 * np.pi * x[:, 0]))[0], 0],
                [0, (2 + 40 * np.pi ** 2 * np.cos(2 * np.pi * x[:, 1]))[0]],
            ]
        )
        n_x = 2  # 'n_x' states
        limits = np.array([[-5, 5], [-5, 5]], dtype=float)  # Boundaries
        obj = 0  # objective value (f(x_min) = obj)
        x_opt = np.array([[0.0, 0.0]])

    elif fun_id == "F6" or fun_id == 4:
        #  F6 function.
        #  Minimum: f(9.6204) = -100.22
        f = lambda x: (x ** 2 + x) * np.cos(x)
        g = lambda x: (2 * x + 1) * np.cos(x) - (x ** 2 + x) * np.sin(x)
        H = (
            lambda x: 2 * np.cos(x)
            - (2 * x + 1) * np.sin(x)
            - (2 * x + 1) * np.sin(x)
            - (x ** 2 + x) * np.cos(x)
        )
        n_x = 1  # 'n_x' states
        limits = np.array([-10.0, 10.0])  # Boundaries
        obj = -100.22  # objective value (f(x_min) = obj)
        x_opt = 9.6204

    elif fun_id == "threehumpcamel" or fun_id == 5:
        #  Three hump camel function
        #  Minimum: f(0,0) = 0
        f = (
            lambda x: 2 * x[:, 0] ** 2
            - 1.05 * x[:, 0] ** 4
            + x[:, 0] ** 6 / 6
            + x[:, 0] * x[:, 1]
            + x[:, 1] ** 2
        )
        g = lambda x: np.array(
            [
                x[:, 0] ** 5 + 4 * x[:, 0] + x[:, 1] - 4.2 * x[:, 0] ** 3,
                x[:, 0] + 2 * x[:, 1],
            ],
            dtype=float,
        )
        H = lambda x: np.array(
            [[(5 * x[:, 0] ** 4 - 12.6 * x[:, 0] ** 2 + 4)[0], 1], [1, 2]]
        )
        n_x = 2  # 'n_x' states
        limits = np.array([[-4.0, 4.0], [-4.0, 4.0]])  # Boundaries
        obj = 0.0
        # objective value (f(x_min) = obj)
        x_opt = np.array([[0.0, 0.0]])

    return f, g, H, n_x, limits, obj, x_opt
