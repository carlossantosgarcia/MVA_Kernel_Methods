import random

import cvxopt
import numpy as np
from scipy import optimize


class KernelSVC:
    """
    Implements kernel support vector binary classifiers.
    """

    def __init__(self, C, kernel=None, epsilon=1e-3):
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.support = None
        self.epsilon = epsilon

    def optimization_solver(self, y, method="scipy", class_weights=None) -> None:
        """Solves the optimization problem for the dual formulation in SVMs.

        Saves the solution vector in self.alpha attribute.

        Args:
            y (np.ndarray): Labels array
            method (str, optional): Optimization module used. Defaults to "scipy".
        """
        random.seed(42)
        cvxopt.setseed(42)
        N = len(y)
        diag_y = np.diag(y)
        yKy = diag_y @ self.K_train @ diag_y

        if class_weights is not None:
            # Balances each class using weights
            assert type(class_weights) in (bool, dict)
            if type(class_weights) == bool:
                if class_weights:
                    # Weights inversely proportional to the size of the classes
                    weights = {}
                    weights[-1] = N / np.sum((y == -1))
                    weights[1] = N / np.sum((y == 1))
                else:
                    # Equal weights for both classes
                    weights = {-1: 1, 1: 1}
            Cvect = weights[-1] * (y == -1) + weights[1] * (y == 1)
            Cvect = self.C * Cvect
        else:
            # No weighting done
            Cvect = self.C * np.ones(N)
        self.Cvect = Cvect

        if method == "scipy":
            # Lagrange dual loss
            def loss(alpha):
                return 0.5 * alpha.T @ yKy @ alpha - np.sum(alpha)

            # Partial derivate of the loss w.r.t. alpha
            def grad_loss(alpha):
                return yKy @ alpha - np.ones(alpha.shape)

            # Constraints on alpha
            fun_eq = lambda alpha: np.dot(alpha, y)  # Equality constraint
            jac_eq = lambda alpha: y  # Gradient of the equality constraint
            fun_ineq = lambda alpha: np.concatenate((alpha, Cvect - alpha))  # Inequality constraints
            jac_ineq = lambda alpha: np.concatenate(
                (np.eye(N), -np.eye(N)),
                axis=0,
            )  # Jacobian of the inequality constraints

            constraints = (
                {"type": "eq", "fun": fun_eq, "jac": jac_eq},
                {
                    "type": "ineq",
                    "fun": fun_ineq,
                    "jac": jac_ineq,
                },
            )

            # Optimization
            np.random.seed(seed=42)
            optRes = optimize.minimize(
                fun=lambda alpha: loss(alpha),
                x0=np.ones(N),
                method="SLSQP",
                jac=lambda alpha: grad_loss(alpha),
                constraints=constraints,
            )
            self.alpha = optRes.x

        elif method == "cvxopt":
            # Quadratic objective
            P = yKy
            q = -np.ones(N)

            # Constraints
            G = np.kron(np.array([[-1.0], [1.0]]), np.eye(N))
            h = np.kron(np.array([0.0, 1.0]), Cvect)
            A = y.reshape(1, -1).astype("float")
            b = np.array([[0.0]]).astype("float")

            # Optimization
            out = cvxopt.solvers.qp(
                P=cvxopt.matrix(P),
                q=cvxopt.matrix(q),
                G=cvxopt.matrix(G),
                h=cvxopt.matrix(h),
                A=cvxopt.matrix(A),
                b=cvxopt.matrix(b),
            )
            self.alpha = np.array(out["x"]).reshape((N,))

    def fit(self, X, y, K_train=None, method="scipy", class_weights=None):
        """_summary_

        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Training labels.
            K_train (np.ndarray, optional): Training kernel matrix. Defaults to None.
            method (str, optional): Optimization module used. Defaults to "scipy".
        """
        assert method in ("scipy", "cvxopt"), "Optimization method not supported !"
        if self.kernel is None:
            # Gram matrix needs to be passed
            assert K_train is not None, "Kernel function or matrix missing !"
            self.K_train = K_train
        else:
            # Gram matrix is computed using the kernel
            assert X is not None, "No data given !"
            self.X_train = X
            self.K_train = self.kernel(X, X)

        # Optimization procedure
        self.optimization_solver(y=y, method=method, class_weights=class_weights)

        # Post-processing of alpha values
        self.alpha = np.where(self.epsilon > self.alpha, 0, self.alpha)
        self.alpha = np.where(self.Cvect - self.epsilon < self.alpha, self.Cvect, self.alpha)

        # Assign the required attributes
        self.f_coeffs = self.alpha * y

        # Points falling within the margin
        idx = np.bitwise_and(self.alpha > 0, self.alpha < self.Cvect).nonzero()

        # Compute relevant quantities to estimate b on points s.t. 0 < alpha < C
        if self.kernel is None:
            self.margin_points = None
            self.K_margin = self.K_train[idx, :]
        else:
            self.margin_points = X[idx]
            self.K_margin = None

        # Offset for the classifier
        margin_values = self.separating_function(X=self.margin_points, K=self.K_margin)
        b_values = y[idx] - margin_values
        self.b = np.mean(b_values)

        # RKHS norm of the function
        self.norm_f = np.sqrt(np.dot(self.alpha, self.K_train @ self.alpha))

    def separating_function(self, X, K=None):
        """Computes f(x) (without offset b) on given data array.

        Args:
            X (np.ndarray): Input data.
            K (np.ndarray, optional): Input kernel matrix. Defaults to None.

        Returns:
            np.ndarray: f(x) values on input data.
        """
        if self.kernel is None:
            assert K is not None, "Kernel function or matrix missing !"
            return K @ self.f_coeffs
        else:
            assert X is not None, "No data given !"
            return self.kernel(X, self.X_train) @ self.f_coeffs

    def predict(self, X, K=None):
        """Computes f(x) + b on the input data.

        Args:
            X (np.ndarray): Input data.
            K (np.ndarray, optional): Input kernel matrix. Defaults to None.

        Returns:
            np.ndarray: Final function on the input data.
        """
        d = self.separating_function(X=X, K=K)
        return d + self.b