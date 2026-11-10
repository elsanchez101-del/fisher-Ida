# -*- coding: utf-8 -*-
"""
comun.py
--------
Parámetros y funciones compartidas por los scripts del caso aplicado (clasificación de riesgo
de acciones del S&P 500). Antes cada script repetía estas funciones; ahora todos las importan
de aquí, así no pueden quedar distintas entre sí.

Convención del caso aplicado: clase 1 = no riesgosa, clase 2 = riesgosa, d = mu2 - mu1 y w
orientado para que un puntaje w^T x alto signifique más riesgo. Es la misma dirección de Fisher
que en src/fisher.py (allí d = mu1 - mu2) con el signo opuesto, y J(-w) = J(w).
"""
import os
import numpy as np
import pandas as pd

# ----------------------------- RUTAS --------------------------------------
# Relativas a esta carpeta: los scripts funcionan desde Spyder (F5) o desde la terminal,
# sin importar la carpeta de trabajo.
BASE        = os.path.dirname(os.path.abspath(__file__))
CARPETA     = os.path.join(BASE, "datos")
RESULTADOS  = os.path.join(BASE, "resultados")
FIGURAS     = os.path.join(BASE, "figuras")
ARCHIVO           = os.path.join(CARPETA, "acciones_dataset_300.csv")     # entrenamiento (300 acciones)
ARCHIVO_COLA      = os.path.join(CARPETA, "cola_usada.txt")               # tickers de la cola ya usados
ARCHIVO_LISTA     = os.path.join(CARPETA, "sp500_lista.csv")              # copia de la lista del S&P 500
ARCHIVO_NUEVAS    = os.path.join(CARPETA, "acciones_validacion.csv")      # acciones de la validación externa

# ----------------------------- MODELO -------------------------------------
VARS        = ["sigma", "beta", "mdd", "de"]
NOMBRES     = ["sigma", "beta", "mdd", "log(1+D/E)"]
USAR_LOG_DE = True          # log(1 + D/E) por la fuerte asimetría del apalancamiento
UMBRAL      = 0.30          # caída máxima > 30 % en el año siguiente => riesgosa (clase 2)
SEMILLA     = 2026


def asegurar_carpetas():
    for c in (CARPETA, RESULTADOS, FIGURAS):
        os.makedirs(c, exist_ok=True)


def a_matriz(df):
    """Matriz X (n x 4) con las variables del modelo."""
    X = df[VARS].to_numpy(dtype=float).copy()
    if USAR_LOG_DE:
        X[:, 3] = np.log1p(X[:, 3])
    return X


def cargar(archivo=ARCHIVO):
    """Lee un CSV de acciones y devuelve (df, X, y)."""
    if not os.path.exists(archivo):
        raise FileNotFoundError(f"No existe {archivo}. Corre primero datos_300.py "
                                f"o copia el CSV en la carpeta datos/.")
    df = pd.read_csv(archivo, index_col="ticker")
    return df, a_matriz(df), df["etiqueta"].to_numpy(dtype=int)


def estandarizar(Xtr, *otros):
    """Media y desviación SOLO del entrenamiento, aplicadas a todos los conjuntos."""
    m, s = Xtr.mean(0), Xtr.std(0, ddof=1)
    return ((Xtr - m) / s,) + tuple((X - m) / s for X in otros)


def separar(y, rng, prop_prueba=0.30):
    """Separación estratificada: respeta la proporción de cada clase."""
    test = []
    for c in (1, 2):
        idx = rng.permutation(np.where(y == c)[0])
        test += list(idx[: int(round(prop_prueba * len(idx)))])
    test = np.array(sorted(test))
    return np.setdiff1d(np.arange(len(y)), test), test


# ----------------------------- DIRECCIONES --------------------------------
def estadisticos(X, y):
    """Medias de cada clase y dispersión intraclase S_W = S_1 + S_2."""
    X1, X2 = X[y == 1], X[y == 2]
    mu1, mu2 = X1.mean(0), X2.mean(0)
    SW = (X1 - mu1).T @ (X1 - mu1) + (X2 - mu2).T @ (X2 - mu2)
    return mu1, mu2, SW


def regularizar(SW, gamma):
    """S_W + lambda I con lambda = gamma * tr(S_W) / p (igual que en src/fisher.py)."""
    return SW + gamma * np.trace(SW) / SW.shape[0] * np.eye(SW.shape[0]) if gamma > 0 else SW


def dir_fisher(X, y, gamma=0.0):
    """Resuelve (S_W + lambda I) w = mu2 - mu1 sin invertir la matriz."""
    mu1, mu2, SW = estadisticos(X, y)
    return np.linalg.solve(regularizar(SW, gamma), mu2 - mu1)


def dir_pca(X):
    """Autovector de la covarianza con el mayor autovalor. No usa las etiquetas."""
    vals, vecs = np.linalg.eigh(np.cov(X, rowvar=False))
    return vecs[:, -1]


def dir_medias(X, y):
    mu1, mu2, _ = estadisticos(X, y)
    return mu2 - mu1


def orientar(w, X, y):
    """w unitario y con el signo tal que la clase 2 (riesgosa) tenga mayor puntaje medio."""
    mu1, mu2, _ = estadisticos(X, y)
    w = np.asarray(w, float) / np.linalg.norm(w)
    return w if w @ (mu2 - mu1) >= 0 else -w


def direcciones(X, y, gamma=0.0):
    """Las direcciones que se comparan, ya orientadas. Pesos iguales no se aprende de los datos."""
    p = X.shape[1]
    return {"LDA (Fisher)": orientar(dir_fisher(X, y, gamma), X, y),
            "PCA (1ª comp.)": orientar(dir_pca(X), X, y),
            "Dif. de medias": orientar(dir_medias(X, y), X, y),
            "Pesos iguales": np.ones(p) / np.sqrt(p)}


def criterio_J(w, X, y):
    """J(w) = (w^T d)^2 / (w^T S_W w), con S_W / (N - 2) para que no dependa de N."""
    mu1, mu2, SW = estadisticos(X, y)
    return float((w @ (mu2 - mu1)) ** 2 / (w @ (SW / (len(y) - 2)) @ w))


def umbral_medio(w, X, y):
    """Punto medio de las medias proyectadas (óptimo con previas y costos iguales)."""
    mu1, mu2, _ = estadisticos(X, y)
    return (w @ mu1 + w @ mu2) / 2


def predecir(w, t, X):
    return np.where(X @ w > t, 2, 1)


# ----------------------------- MÉTRICAS -----------------------------------
def metricas(real, pred):
    """Matriz de confusión (filas = real 1, 2; columnas = predicho 1, 2) y medidas derivadas."""
    cm = np.array([[np.sum((real == i) & (pred == j)) for j in (1, 2)] for i in (1, 2)])
    rec1 = cm[0, 0] / max(cm[0].sum(), 1)
    rec2 = cm[1, 1] / max(cm[1].sum(), 1)
    prec2 = cm[1, 1] / cm[:, 1].sum() if cm[:, 1].sum() > 0 else np.nan
    return dict(aciertos=int(np.trace(cm)), exactitud=np.trace(cm) / cm.sum(),
                balanceada=(rec1 + rec2) / 2, riesgosas_detectadas=rec2, precision_riesgosa=prec2,
                etiq_como_1=int(cm[:, 0].sum()), etiq_como_2=int(cm[:, 1].sum()), cm=cm)


def auc(puntaje, y):
    """Área bajo la curva ROC: probabilidad de que una riesgosa tenga mayor puntaje que una
    no riesgosa (empates cuentan 1/2). No depende del umbral, solo de la dirección."""
    p2, p1 = puntaje[y == 2], puntaje[y == 1]
    mayor = (p2[:, None] > p1[None, :]).sum() + 0.5 * (p2[:, None] == p1[None, :]).sum()
    return float(mayor / (len(p2) * len(p1)))
