"""
Regularization and Overfitting
Machine Learning Foundations Homework

Demonstrates underfitting, overfitting, and L2 regularization (Ridge)
via polynomial regression on a noisy sinusoidal dataset.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(n_train=20, n_test=100, noise=0.2, seed=42):
    """
    Sample from y = sin(2π x) + ε,  x ~ Uniform[0, 1].

    Returns
    -------
    x_train, y_train : ndarray shape (n_train,)
    x_test,  y_test  : ndarray shape (n_test,)
    """
    rng = np.random.default_rng(seed)

    x_train = rng.uniform(0, 1, n_train)
    y_train = np.sin(2 * np.pi * x_train) + rng.normal(0, noise, n_train)

    x_test = np.linspace(0, 1, n_test)
    y_test = np.sin(2 * np.pi * x_test) + rng.normal(0, noise, n_test)

    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def polynomial_features(x, degree):
    """
    Build a Vandermonde-style design matrix with a bias column.

    Parameters
    ----------
    x      : 1-D array of shape (n,)
    degree : int, maximum polynomial degree

    Returns
    -------
    X : ndarray of shape (n, degree + 1)
        Columns are [1, x, x², …, x^degree]
    """
    return np.column_stack([x ** d for d in range(degree + 1)])


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_least_squares(X, y):
    """
    Ordinary Least Squares via the normal equations.

    w* = (X^T X)^{-1} X^T y

    Returns
    -------
    w : ndarray of shape (n_features,)
    """
    return np.linalg.pinv(X.T @ X) @ X.T @ y


def fit_ridge(X, y, lam):
    """
    Ridge Regression (L2-regularised least squares).

    w* = (X^T X + λ I)^{-1} X^T y

    The bias term (first column) is regularised along with the weights;
    for homework purposes this is fine. In production, centre the target
    and exclude the intercept from the penalty.

    Parameters
    ----------
    X   : design matrix, shape (n, p)
    y   : targets,       shape (n,)
    lam : regularisation strength λ ≥ 0

    Returns
    -------
    w : ndarray of shape (p,)
    """
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)


# ---------------------------------------------------------------------------
# Prediction and evaluation
# ---------------------------------------------------------------------------

def predict(X, w):
    """Return X @ w."""
    return X @ w


def mean_squared_error(y_true, y_pred):
    """MSE = mean of squared residuals."""
    residuals = y_true - y_pred
    return float(np.mean(residuals ** 2))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    x_train, y_train, x_test, y_test = generate_data()

    # ------------------------------------------------------------------ #
    # Part 1 – Polynomial degree vs. train / test error (no regularisation)
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Part 1: Model Complexity (OLS, varying degree)")
    print("=" * 60)
    print(f"{'Degree':>6}  {'Train MSE':>10}  {'Test MSE':>10}  Verdict")
    print("-" * 48)

    degrees = [1, 3, 5, 9, 15]
    results_ols = {}

    for deg in degrees:
        X_tr = polynomial_features(x_train, deg)
        X_te = polynomial_features(x_test,  deg)

        w = fit_least_squares(X_tr, y_train)

        train_mse = mean_squared_error(y_train, predict(X_tr, w))
        test_mse  = mean_squared_error(y_test,  predict(X_te, w))
        results_ols[deg] = (train_mse, test_mse)

        if deg <= 2:
            verdict = "underfit"
        elif deg <= 5:
            verdict = "good fit"
        else:
            verdict = "overfit"

        print(f"{deg:>6}  {train_mse:>10.4f}  {test_mse:>10.4f}  {verdict}")

    # ------------------------------------------------------------------ #
    # Part 2 – Ridge regularisation on the high-degree model
    # ------------------------------------------------------------------ #
    print()
    print("=" * 60)
    print("Part 2: Ridge Regularisation (degree = 9)")
    print("=" * 60)
    print(f"{'Lambda':>12}  {'Train MSE':>10}  {'Test MSE':>10}")
    print("-" * 40)

    degree = 9
    X_tr = polynomial_features(x_train, degree)
    X_te = polynomial_features(x_test,  degree)

    lambdas = [0.0, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0]

    for lam in lambdas:
        if lam == 0.0:
            w = fit_least_squares(X_tr, y_train)
        else:
            w = fit_ridge(X_tr, y_train, lam)

        train_mse = mean_squared_error(y_train, predict(X_tr, w))
        test_mse  = mean_squared_error(y_test,  predict(X_te, w))

        print(f"{lam:>12.6f}  {train_mse:>10.4f}  {test_mse:>10.4f}")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print()
    print("=" * 60)
    print("Key observations")
    print("=" * 60)
    print("• Low degree  → high train AND test error (underfitting).")
    print("• High degree → low train error, high test error (overfitting).")
    print("• Ridge (λ > 0) shrinks weights, reducing test error on")
    print("  the degree-9 model; very large λ re-introduces bias.")


if __name__ == "__main__":
    main()
