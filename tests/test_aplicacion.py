"""Pruebas del caso aplicado (aplicacion/comun.py). No necesitan internet ni los datos reales."""
import sys, pathlib
raiz = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(raiz / "src"))
sys.path.insert(0, str(raiz / "aplicacion"))
import numpy as np
import comun as c
from fisher import fisher_direction


def datos(seed=0, n=120):
    rng = np.random.default_rng(seed)
    S = np.array([[1, .6, .4, 0], [.6, 1, .3, 0], [.4, .3, 1, .1], [0, 0, .1, 1]])
    y = np.r_[np.ones(n, int), 2 * np.ones(n // 2, int)]
    X = rng.multivariate_normal(np.zeros(4), S, len(y)) + (y == 2)[:, None] * np.array([.8, .6, .2, 0])
    return X, y


def test_misma_direccion_que_src_fisher():
    """La dirección del caso aplicado es la de src/fisher.py con el signo opuesto (d = mu2 - mu1)."""
    X, y = datos()
    w_app = c.orientar(c.dir_fisher(X, y), X, y)
    w_src = fisher_direction(X, y)                # usa d = mu1 - mu2
    assert np.isclose(abs(w_app @ w_src), 1, atol=1e-10)
    for g in (0.1, 3.0):                          # también con regularización (mismo lambda)
        w_app = c.orientar(c.dir_fisher(X, y, g), X, y)
        assert np.isclose(abs(w_app @ fisher_direction(X, y, gamma=g)), 1, atol=1e-10)


def test_orientacion_riesgosas_con_puntaje_mayor():
    X, y = datos(1)
    for nombre, w in c.direcciones(X, y).items():
        assert np.isclose(np.linalg.norm(w), 1)
        if nombre != "Pesos iguales":             # pesos iguales es fijo, no se orienta
            mu1, mu2, _ = c.estadisticos(X, y)
            assert w @ mu2 >= w @ mu1


def test_estandarizacion_solo_con_entrenamiento():
    """La prueba se transforma con la media y la desviación del entrenamiento, no con las suyas."""
    rng = np.random.default_rng(2)
    Xtr, Xte = rng.normal(5, 2, (50, 4)), rng.normal(9, 3, (20, 4))
    Ztr, Zte = c.estandarizar(Xtr, Xte)
    assert np.allclose(Ztr.mean(0), 0) and np.allclose(Ztr.std(0, ddof=1), 1)
    assert np.allclose(Zte, (Xte - Xtr.mean(0)) / Xtr.std(0, ddof=1))
    assert not np.allclose(Zte.mean(0), 0)


def test_separacion_estratificada():
    X, y = datos(3)
    tr, te = c.separar(y, np.random.default_rng(0), 0.3)
    assert len(np.intersect1d(tr, te)) == 0 and len(tr) + len(te) == len(y)
    for k in (1, 2):
        assert abs(np.mean(y[te] == k) - np.mean(y == k)) < 0.02


def test_metricas_y_auc():
    real = np.array([1, 1, 1, 2, 2, 2, 2])
    pred = np.array([1, 2, 1, 2, 2, 1, 2])
    m = c.metricas(real, pred)
    assert m["cm"].tolist() == [[2, 1], [1, 3]]
    assert m["aciertos"] == 5 and np.isclose(m["balanceada"], (2 / 3 + 3 / 4) / 2)
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(4)
    s, yy = rng.normal(size=60).round(1), rng.integers(1, 3, 60)     # con empates
    assert np.isclose(c.auc(s, yy), roc_auc_score(yy == 2, s))


def test_J_maximo_en_fisher():
    X, y = datos(5)
    Jf = c.criterio_J(c.orientar(c.dir_fisher(X, y), X, y), X, y)
    rng = np.random.default_rng(6)
    assert all(c.criterio_J(rng.normal(size=4), X, y) <= Jf + 1e-9 for _ in range(300))
