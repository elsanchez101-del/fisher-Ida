"""Discriminante lineal de Fisher implementado desde cero (numpy)."""
import numpy as np


def class_stats(X, y):
    """Medias y matrices de dispersión por clase. Devuelve (clases, mus, S_list, n_list)."""
    classes = np.unique(y)
    mus, Ss, ns = [], [], []
    for c in classes:
        Xc = X[y == c]
        mu = Xc.mean(axis=0)
        D = Xc - mu
        mus.append(mu); Ss.append(D.T @ D); ns.append(len(Xc))
    return classes, np.array(mus), Ss, np.array(ns)


def within_scatter(X, y):
    _, _, Ss, _ = class_stats(X, y)
    return sum(Ss)


def regularize(SW, gamma):
    """S_{W,lambda} = S_W + lambda I, con lambda = gamma*tr(S_W)/p (si tr=0, lambda=gamma)."""
    p = SW.shape[0]
    tr = np.trace(SW)
    lam = gamma * tr / p if tr > 0 else gamma
    return SW + lam * np.eye(p)


def fisher_direction(X, y, gamma=0.0):
    """Dos clases: resuelve (S_W + lam I) w = mu1 - mu2 (sin invertir). w normalizado a norma 1."""
    classes, mus, Ss, _ = class_stats(X, y)
    assert len(classes) == 2
    SW = sum(Ss)
    if gamma > 0:
        SW = regularize(SW, gamma)
    d = mus[0] - mus[1]
    try:
        w = np.linalg.solve(SW, d)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(SW) @ d
    return w / np.linalg.norm(w)


def fisher_J(w, X, y):
    """Criterio de Fisher J(w) = (w^T(mu1-mu2))^2 / (w^T S_W w)."""
    _, mus, Ss, _ = class_stats(X, y)
    SW = sum(Ss)
    d = mus[0] - mus[1]
    return float((w @ d) ** 2 / (w @ SW @ w))


class FisherLDA:
    """Clasificador de 2 clases: proyecta y umbral en el punto medio de medias proyectadas."""

    def __init__(self, gamma=0.0):
        self.gamma = gamma

    def fit(self, X, y):
        self.classes_, mus, _, _ = class_stats(X, y)
        self.w_ = fisher_direction(X, y, self.gamma)
        m1, m2 = self.w_ @ mus[0], self.w_ @ mus[1]
        self.t_ = (m1 + m2) / 2
        self.sign_ = 1 if m1 < m2 else -1  # orientación: clase 1 a un lado
        return self

    def project(self, X):
        return X @ self.w_

    def predict(self, X):
        y = self.project(X)
        c1, c2 = self.classes_
        if self.sign_ == 1:
            return np.where(y < self.t_, c1, c2)
        return np.where(y > self.t_, c1, c2)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))


def multiclass_fisher(X, y, r=None, gamma=0.0):
    """K clases: problema generalizado S_B w = lam S_W w (scipy.linalg.eigh)."""
    from scipy.linalg import eigh
    classes, mus, Ss, ns = class_stats(X, y)
    mu = (ns[:, None] * mus).sum(0) / ns.sum()
    SW = sum(Ss)
    if gamma > 0:
        SW = regularize(SW, gamma)
    SB = sum(n * np.outer(m - mu, m - mu) for n, m in zip(ns, mus))
    vals, vecs = eigh(SB, SW)
    idx = np.argsort(vals)[::-1]
    r = r or min(X.shape[1], len(classes) - 1)
    return vals[idx][:r], vecs[:, idx][:, :r]
