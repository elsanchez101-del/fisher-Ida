# -*- coding: utf-8 -*-
"""
tres_clases.py
--------------
Fisher con TRES clases de riesgo y el plano span(w1, w2).

Clases según la caída máxima del año siguiente (mdd_futuro):
    bajo  : mdd_futuro <= CORTE_BAJO
    medio : CORTE_BAJO < mdd_futuro <= UMBRAL (0.30)
    alto  : mdd_futuro > UMBRAL                 (coincide con "riesgosa" del caso de dos clases)

Con K clases:  S_W = sum_k S_k,  S_B = sum_k N_k (mu_k - mu)(mu_k - mu)^T.
Como sum_k N_k (mu_k - mu) = 0, rango(S_B) <= K - 1 = 2: el problema generalizado
    S_B w = lambda S_W w
tiene a lo sumo DOS autovalores no nulos. Sus autovectores w1, w2 generan el plano
span(w1, w2) en el que las tres clases quedan lo más separadas posible. Así se ve en dos
dimensiones lo que pasa en R^4, sin elegir el plano a mano.

Se compara con el plano de PCA (dos primeras componentes, sin etiquetas). La clasificación usa
el centroide más cercano en cada plano (en el de Fisher las coordenadas ya están blanqueadas:
W^T S_W W = I) y se mide con las mismas 20 particiones de la validación interna.

USO: F5 en Spyder o `python aplicacion/tres_clases.py`. Usa la columna mdd_futuro del CSV.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh

import comun as c

CORTE_BAJO  = 0.20
N_REP       = 20
PROP_PRUEBA = 0.30
NOMBRE_CL   = {1: "Bajo", 2: "Medio", 3: "Alto"}
COLOR_CL    = {1: "tab:blue", 2: "tab:orange", 3: "tab:red"}


def clases(df):
    f = df["mdd_futuro"].to_numpy()
    return np.where(f > c.UMBRAL, 3, np.where(f > CORTE_BAJO, 2, 1))


def fisher_multiclase(X, y):
    """Autovalores y autovectores de S_B w = lambda S_W w, de mayor a menor."""
    mu = X.mean(0)
    SW = np.zeros((X.shape[1],) * 2)
    SB = np.zeros_like(SW)
    for k in np.unique(y):
        Xk = X[y == k]
        mk = Xk.mean(0)
        SW += (Xk - mk).T @ (Xk - mk)
        SB += len(Xk) * np.outer(mk - mu, mk - mu)
    vals, vecs = eigh(SB, SW)                      # vecs cumple vecs^T S_W vecs = I
    orden = np.argsort(vals)[::-1]
    K = len(np.unique(y))
    vecs = vecs[:, orden] * np.sqrt(len(y) - K)    # escala: W^T Sigma W = I, con Sigma = S_W/(N-K)
    alto, bajo = X[y == y.max()].mean(0), X[y == y.min()].mean(0)
    for j in range(vecs.shape[1]):                 # signo: la clase alta con mayor puntaje
        if vecs[:, j] @ (alto - bajo) < 0:
            vecs[:, j] = -vecs[:, j]
    return vals[orden], vecs, SB


def plano_pca(X):
    vals, vecs = np.linalg.eigh(np.cov(X, rowvar=False))
    return vecs[:, ::-1][:, :2]


def centroide_cercano(Ztr, ytr, Zte):
    ks = np.unique(ytr)
    cent = np.array([Ztr[ytr == k].mean(0) for k in ks])
    dist = ((Zte[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
    return ks[dist.argmin(1)]


def balanceada(real, pred):
    return np.mean([np.mean(pred[real == k] == k) for k in np.unique(real)])


def main():
    pd.set_option("display.width", 200)
    c.asegurar_carpetas()
    df, X, _ = c.cargar()
    y3 = clases(df)
    print("Clases:", {NOMBRE_CL[k]: int(np.sum(y3 == k)) for k in (1, 2, 3)})
    (Xs,) = c.estandarizar(X)

    vals, W, SB = fisher_multiclase(Xs, y3)
    rango = np.linalg.matrix_rank(SB, tol=1e-9 * np.abs(SB).max())
    print(f"\nrango(S_B) = {rango} (a lo sumo K - 1 = 2)")
    print("Autovalores de S_B w = λ S_W w:", np.round(vals, 4))
    print("Proporción de separación en el plano span(w1, w2):", round(vals[:2].sum() / vals.sum(), 4))
    Wd = pd.DataFrame(W[:, :2] / np.linalg.norm(W[:, :2], axis=0), index=c.NOMBRES, columns=["w1", "w2"])
    print("\nDirecciones w1 y w2 (variables estandarizadas, normalizadas a norma 1):")
    print(Wd.round(3))
    Wd.to_csv(os.path.join(c.RESULTADOS, "tres_clases_direcciones.csv"))

    # ---------------- validación interna: centroide más cercano -------------
    rng = np.random.default_rng(c.SEMILLA)
    filas = []
    for rep in range(N_REP):
        tr, te = c.separar(np.where(y3 == 3, 2, 1), rng, PROP_PRUEBA)   # misma regla de partición
        Xtr, Xte = c.estandarizar(X[tr], X[te])
        _, Wtr, _ = fisher_multiclase(Xtr, y3[tr])
        P = plano_pca(Xtr)
        base = np.bincount(y3[te]).max() / len(te)
        filas.append(dict(rep=rep,
                          fisher_2D=balanceada(y3[te], centroide_cercano(Xtr @ Wtr[:, :2], y3[tr], Xte @ Wtr[:, :2])),
                          pca_2D=balanceada(y3[te], centroide_cercano(Xtr @ P, y3[tr], Xte @ P)),
                          linea_base_exactitud=base))
    res = pd.DataFrame(filas)
    print(f"\nExactitud balanceada en PRUEBA con 3 clases ({N_REP} particiones; azar = 0.333):")
    print(res[["fisher_2D", "pca_2D"]].agg(["mean", "std"]).round(3))
    res.to_csv(os.path.join(c.RESULTADOS, "tres_clases_validacion.csv"), index=False)

    # ------------------------------ figura ---------------------------------
    P = plano_pca(Xs)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, Z, titulo, ejes in ((axs[0], Xs @ W[:, :2], "span(w1, w2): plano de Fisher", ("w1", "w2")),
                                (axs[1], Xs @ P, "Plano de PCA (sin etiquetas)", ("PC1", "PC2"))):
        for k in (1, 2, 3):
            ax.scatter(Z[y3 == k, 0], Z[y3 == k, 1], s=22, alpha=.6, c=COLOR_CL[k], label=f"{NOMBRE_CL[k]} ({np.sum(y3 == k)})")
            m = Z[y3 == k].mean(0)
            ax.scatter(*m, s=180, marker="X", c=COLOR_CL[k], edgecolors="k", linewidths=1)
        ax.set_xlabel(ejes[0]); ax.set_ylabel(ejes[1]); ax.set_title(titulo); ax.legend(fontsize=8)
    fig.suptitle("Tres clases de riesgo: las 4 variables proyectadas en un plano (X = media de cada clase)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(c.FIGURAS, "tres_clases_span.png"), dpi=150)
    plt.show()
    return vals, W, res


if __name__ == "__main__":
    autovalores, direcciones, resultados = main()
