import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SK
from fisher import *

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
