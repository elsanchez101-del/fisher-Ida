# -*- coding: utf-8 -*-
"""
tres_clases.py
--------------
Fisher con TRES clases de riesgo y el plano span(w1, w2).

Clases según la caída máxima del año siguiente (mdd_futuro):
    bajo  : mdd_futuro <= 0.20
    medio : 0.20 < mdd_futuro <= 0.30
    alto  : mdd_futuro > 0.30        (coincide con "riesgosa" del caso de dos clases)

Con K clases:  S_W = sum_k S_k   y   S_B = sum_k N_k (mu_k - mu)(mu_k - mu)^T.
Como sum_k N_k (mu_k - mu) = 0, los K vectores (mu_k - mu) son linealmente dependientes y
rango(S_B) <= K - 1 = 2. Por eso el problema generalizado
    S_B w = lambda S_W w
tiene a lo sumo DOS autovalores distintos de cero. Sus autovectores w1 y w2 generan el plano
span(w1, w2) donde las tres clases quedan lo más separadas posible: vemos en 2 dimensiones lo
que pasa en R^4. Se compara con el plano de PCA (dos primeras componentes, sin etiquetas).

USO: F5 en Spyder o `python aplicacion/tres_clases.py`. Usa la columna mdd_futuro del CSV.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh

import comun as c

CORTE_BAJO = 0.20
NOMBRE_CL  = {1: "Bajo", 2: "Medio", 3: "Alto"}
COLOR_CL   = {1: "tab:blue", 2: "tab:orange", 3: "tab:red"}


def clases(df):
    """1 = bajo, 2 = medio, 3 = alto, según la caída máxima del año siguiente."""
    f = df["mdd_futuro"].to_numpy()
    return np.where(f > c.UMBRAL, 3, np.where(f > CORTE_BAJO, 2, 1))


def fisher_multiclase(X, y):
    """S_W, S_B y la solución de S_B w = lambda S_W w (autovalores de mayor a menor;
    cada autovector normalizado a norma 1 y orientado para que 'alto' tenga mayor puntaje)."""
    mu = X.mean(0)
    p = X.shape[1]
    SW, SB = np.zeros((p, p)), np.zeros((p, p))
    for k in np.unique(y):
        Xk = X[y == k]
        mk = Xk.mean(0)
        SW += (Xk - mk).T @ (Xk - mk)                 # dispersión dentro de la clase k
        SB += len(Xk) * np.outer(mk - mu, mk - mu)    # dispersión entre clases
    vals, vecs = eigh(SB, SW)                         # problema de autovalores generalizado
    orden = np.argsort(vals)[::-1]
    vals, vecs = vals[orden], vecs[:, orden]
    vecs = vecs / np.linalg.norm(vecs, axis=0)
    alto, bajo = X[y == 3].mean(0), X[y == 1].mean(0)
    for j in range(p):
        if vecs[:, j] @ (alto - bajo) < 0:
            vecs[:, j] = -vecs[:, j]
    return vals, vecs, SB


def plano_pca(X):
    """Las dos primeras componentes principales."""
    vals, vecs = np.linalg.eigh(np.cov(X, rowvar=False))
    return vecs[:, ::-1][:, :2]


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
    Wd = pd.DataFrame(W[:, :2], index=c.NOMBRES, columns=["w1", "w2"])
    print("\nDirecciones w1 y w2 (variables estandarizadas, norma 1):")
    print(Wd.round(3))
    Wd.to_csv(os.path.join(c.RESULTADOS, "tres_clases_direcciones.csv"))

    P = plano_pca(Xs)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, Z, titulo, ejes in ((axs[0], Xs @ W[:, :2], "span(w1, w2): plano de Fisher", ("w1", "w2")),
                                (axs[1], Xs @ P, "Plano de PCA (sin etiquetas)", ("PC1", "PC2"))):
        for k in (1, 2, 3):
            ax.scatter(Z[y3 == k, 0], Z[y3 == k, 1], s=22, alpha=.6, c=COLOR_CL[k],
                       label=f"{NOMBRE_CL[k]} ({np.sum(y3 == k)})")
            ax.scatter(*Z[y3 == k].mean(0), s=180, marker="X", c=COLOR_CL[k], edgecolors="k", linewidths=1)
        lo, hi = np.percentile(Z, [1, 99], axis=0)        # ejes sin dejarse estirar por un valor atípico
        margen = 0.15 * (hi - lo)
        ax.set_xlim(lo[0] - margen[0], hi[0] + margen[0]); ax.set_ylim(lo[1] - margen[1], hi[1] + margen[1])
        ax.set_xlabel(ejes[0]); ax.set_ylabel(ejes[1]); ax.set_title(titulo); ax.legend(fontsize=8)
    fig.suptitle("Tres clases de riesgo: las 4 variables proyectadas en un plano (X = media de cada clase)",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(c.FIGURAS, "tres_clases_span.png"), dpi=150)
    plt.show()
    return vals, W


if __name__ == "__main__":
    autovalores, direcciones = main()
