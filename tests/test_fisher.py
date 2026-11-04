import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SK
from fisher import (FisherLDA, class_stats, fisher_J, fisher_direction,
                    multiclass_fisher, theoretical_accuracy, within_scatter)

C1 = np.array([[1,1],[2,1],[2,2]], float); C2 = np.array([[4,3],[5,3],[5,4]], float)
X = np.vstack([C1, C2]); y = np.array([0,0,0,1,1,1])

def test_manual_example():
    _, mus, Ss, _ = class_stats(X, y)
    assert np.allclose(mus[0], [5/3, 4/3]) and np.allclose(mus[1], [14/3, 10/3])
    assert np.allclose(Ss[0], [[2/3,1/3],[1/3,2/3]]) and np.allclose(Ss[1], Ss[0])
    SW = within_scatter(X, y)
    w = np.linalg.solve(SW, mus[0]-mus[1])
    assert np.allclose(w, [-2, -0.5])
    w = fisher_direction(X, y); assert np.allclose(w/w[1], [4, 1])
    m = FisherLDA().fit(X, y)
    assert np.allclose(np.sort(X @ np.array([4,1])), [5,9,10,19,23,24])
    assert np.isclose(abs(m.t_) * np.linalg.norm([4,1]), 15)
    assert m.score(X, y) == 1.0

def test_scale_and_sign_invariance():
    w = fisher_direction(X, y)
    assert np.isclose(fisher_J(w, X, y), fisher_J(-3*w, X, y))

def test_max_value_is_mahalanobis():
    _, mus, Ss, _ = class_stats(X, y); SW = sum(Ss); d = mus[0]-mus[1]
    assert np.isclose(fisher_J(fisher_direction(X, y), X, y), d @ np.linalg.solve(SW, d))

def test_J_is_maximal_vs_random():
    rng = np.random.default_rng(0)
    Jb = fisher_J(fisher_direction(X, y), X, y)
    assert all(fisher_J(rng.normal(size=2), X, y) <= Jb + 1e-9 for _ in range(500))

def test_vs_sklearn_direction():
    rng = np.random.default_rng(1)
    S = np.array([[1,.8],[.8,1]])
    Xs = np.vstack([rng.multivariate_normal([0,0],S,100), rng.multivariate_normal([1,0],S,100)])
    ys = np.r_[np.zeros(100), np.ones(100)].astype(int)
    w = fisher_direction(Xs, ys); w2 = SK(solver="lsqr").fit(Xs, ys).coef_[0]; w2 /= np.linalg.norm(w2)
    assert np.isclose(abs(w @ w2), 1, atol=1e-8)
    assert (FisherLDA().fit(Xs, ys).predict(Xs) == SK().fit(Xs, ys).predict(Xs)).mean() > 0.99

def test_scatter_psd_symmetric():
    SW = within_scatter(X, y)
    assert np.allclose(SW, SW.T) and np.linalg.eigvalsh(SW).min() > -1e-12

def test_rank_SB_multiclass():
    rng = np.random.default_rng(2)
    Xm = np.vstack([rng.normal(m, 1, (30, 4)) for m in ([0]*4, [2,0,0,0], [0,2,0,0])])
    ym = np.repeat([0,1,2], 30)
    vals, _ = multiclass_fisher(Xm, ym, r=4)
    assert np.sum(vals > 1e-9) == 2

def test_regularization_singular():
    rng = np.random.default_rng(3)
    Xs = rng.normal(size=(6, 10)); ys = np.array([0,0,0,1,1,1]); Xs[3:] += 1
    w = fisher_direction(Xs, ys, gamma=0.1)
    assert np.all(np.isfinite(w))

# ---------- pruebas añadidas ----------

def test_equal_means_raises():
    """Si mu1 = mu2 no hay dirección discriminante (Observación 5.2)."""
    Xe = np.array([[0,0],[2,2],[0,2],[2,0]], float); ye = np.array([0,0,1,1])
    with pytest.raises(ValueError, match="medias"):
        fisher_direction(Xe, ye)

def test_large_gamma_tends_to_mean_difference():
    """Con lambda muy grande, (S_W + lambda I)^{-1} d apunta en la dirección de d."""
    _, mus, _, _ = class_stats(X, y); d = mus[0] - mus[1]
    w_big = fisher_direction(X, y, gamma=1e8)
    assert np.isclose(abs(w_big @ d) / np.linalg.norm(d), 1, atol=1e-6)
    w0 = fisher_direction(X, y)
    assert abs(w0 @ d) / np.linalg.norm(d) < 0.95   # sin regularizar, Fisher corrige la dirección

def test_label_swap_same_classification():
    """Intercambiar las etiquetas invierte w pero no cambia la partición de los puntos."""
    rng = np.random.default_rng(4)
    S = np.array([[4, 1.8], [1.8, 1]])
    Xs = np.vstack([rng.multivariate_normal([0,0],S,50), rng.multivariate_normal([1,0],S,50)])
    ys = np.r_[np.zeros(50), np.ones(50)].astype(int)
    p1 = FisherLDA().fit(Xs, ys).predict(Xs)
    p2 = FisherLDA().fit(Xs, 1 - ys).predict(Xs)
    assert np.array_equal(p1, 1 - p2)
    assert np.allclose(fisher_direction(Xs, ys), -fisher_direction(Xs, 1 - ys))

def test_J_undefined_in_kernel():
    """En una dirección del núcleo de S_W el cociente no está definido."""
    Xk = np.array([[0,0],[1,0],[0,1],[1,1]], float); yk = np.array([0,0,1,1])  # S_W = diag(1, 0)
    with pytest.raises(ValueError):
        fisher_J(np.array([0., 1.]), Xk, yk)

def test_singular_solve_warns_and_pinv_is_finite():
    """Con N < p, solve avisa que S_W es singular; la pseudoinversa da un vector finito."""
    rng = np.random.default_rng(5)
    Xs = rng.normal(size=(6, 10)); ys = np.array([0,0,0,1,1,1]); Xs[3:] += 1
    with pytest.warns(RuntimeWarning, match="singular"):
        fisher_direction(Xs, ys)
    assert np.all(np.isfinite(fisher_direction(Xs, ys, method="pinv")))

def test_theoretical_accuracy():
    """Phi(delta / 2 sigma): caso conocido y Fisher (Bayes) como máximo frente a otras direcciones."""
    from scipy.stats import norm
    I = np.eye(2)
    assert np.isclose(theoretical_accuracy([1, 0], [0, 0], [2, 0], I), norm.cdf(1))
    S = np.array([[4, 1.8], [1.8, 1]]); m1, m2 = np.array([0., 0.]), np.array([1., 0.])
    wF = np.linalg.solve(S, m1 - m2); Delta = np.sqrt((m1 - m2) @ wF)
    best = theoretical_accuracy(wF, m1, m2, S)
    assert np.isclose(best, norm.cdf(Delta / 2))
    rng = np.random.default_rng(6)
    assert all(theoretical_accuracy(rng.normal(size=2), m1, m2, S) <= best + 1e-12 for _ in range(500))
