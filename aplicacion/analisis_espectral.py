# -*- coding: utf-8 -*-
"""
analisis_espectral.py
---------------------
¿Por qué Fisher no supera a direcciones más simples en datos reales? Dos análisis con álgebra lineal.

1. DESCOMPOSICIÓN ESPECTRAL DE w.
   Con Sigma = S_W / (N - 2) = U diag(lambda_i) U^T (teorema espectral) y d = mu2 - mu1,
       w = Sigma^{-1} d = sum_i (u_i^T d / lambda_i) u_i .
   Cada autodirección u_i entra con peso (u_i^T d) / lambda_i: Fisher DIVIDE entre lambda_i.
   Si d se estima con error e (Cov(e) ~ k Sigma, k = 1/N1 + 1/N2), el error del coeficiente i
   tiene desviación sqrt(k lambda_i) / lambda_i = sqrt(k / lambda_i): las direcciones con
   lambda_i pequeño amplifican el ruido. La razón señal/ruido de cada componente es
       SNR_i = |u_i^T d| / sqrt(k lambda_i).

2. CAMINO DE REGULARIZACIÓN.
   w_gamma = (S_W + lambda I)^{-1} d, lambda = gamma tr(S_W)/p. Con gamma = 0 es Fisher; con
   gamma -> infinito, w_gamma -> d (diferencia de medias). Se mide la exactitud balanceada y el
   AUC en PRUEBA con las mismas 20 particiones de escenarios_lda_pca.py.

USO: F5 en Spyder o `python aplicacion/analisis_espectral.py`. Solo usa el entrenamiento de 300
acciones (validación interna); no mira la validación externa.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import comun as c

N_REP       = 20
PROP_PRUEBA = 0.30
GAMMAS      = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0]


def descomposicion(X, y):
    """Autovalores y autovectores de Sigma = S_W/(N-2), coeficientes de w y señal/ruido."""
    mu1, mu2, SW = c.estadisticos(X, y)
    N1, N2 = np.sum(y == 1), np.sum(y == 2)
    Sigma = SW / (len(y) - 2)
    lam, U = np.linalg.eigh(Sigma)                  # orden creciente
    lam, U = lam[::-1], U[:, ::-1]                  # mayor a menor
    d = mu2 - mu1
    proy = U.T @ d                                  # u_i^T d
    coef = proy / lam                               # peso de u_i en w = Sigma^{-1} d
    k = 1 / N1 + 1 / N2
    snr = np.abs(proy) / np.sqrt(k * lam)
    return lam, U, d, proy, coef, snr, Sigma


def main():
    pd.set_option("display.width", 200)
    c.asegurar_carpetas()
    df, X, y = c.cargar()
    (Xs,) = c.estandarizar(X)

    # ---------------------------- 1) ESPECTRO ------------------------------
    lam, U, d, proy, coef, snr, Sigma = descomposicion(Xs, y)
    w = U @ coef
    assert np.allclose(w, np.linalg.solve(Sigma, d))          # la suma reconstruye Sigma^{-1} d
    tabla = pd.DataFrame({"lambda_i": lam, "u_i^T d": proy, "peso (u_i^T d)/lambda_i": coef,
                          "% de ||w||^2": 100 * coef ** 2 / np.sum(coef ** 2), "señal/ruido": snr},
                         index=[f"u{i + 1}" for i in range(len(lam))])
    autovec = pd.DataFrame(U, index=c.NOMBRES, columns=tabla.index)
    print("=== Descomposición espectral de w = Sigma^{-1} d (variables estandarizadas, 300 acciones) ===")
    print(tabla.round(3))
    print("\nAutovectores u_i (columnas):")
    print(autovec.round(3))
    print(f"\nNúmero de condición de Sigma: {lam[0] / lam[-1]:.1f}")
    tabla.to_csv(os.path.join(c.RESULTADOS, "espectro_w.csv"))
    autovec.to_csv(os.path.join(c.RESULTADOS, "espectro_autovectores.csv"))

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    pos = np.arange(len(lam))
    axs[0].bar(pos, lam, color="tab:gray"); axs[0].set_title("Autovalores λᵢ de Σ")
    axs[1].bar(pos, np.abs(coef), color="tab:blue"); axs[1].set_title("|peso| de uᵢ en w = |uᵢᵀd| / λᵢ")
    axs[2].bar(pos, snr, color="tab:orange"); axs[2].axhline(2, color="k", ls="--", lw=1)
    axs[2].set_title("Señal/ruido de cada componente")
    for ax in axs:
        ax.set_xticks(pos); ax.set_xticklabels(tabla.index)
    fig.suptitle("Fisher divide entre λᵢ: las direcciones de λ pequeño pesan más y son más ruidosas", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(c.FIGURAS, "espectro_w.png"), dpi=150)

    # ---------------------- 2) CAMINO DE REGULARIZACIÓN ---------------------
    rng = np.random.default_rng(c.SEMILLA)            # mismas particiones que escenarios_lda_pca.py
    filas = []
    for rep in range(N_REP):
        tr, te = c.separar(y, rng, PROP_PRUEBA)
        Xtr, Xte = c.estandarizar(X[tr], X[te])
        ytr, yte = y[tr], y[te]
        ref = c.direcciones(Xtr, ytr)
        for g in GAMMAS:
            wg = c.orientar(c.dir_fisher(Xtr, ytr, g), Xtr, ytr)
            m = c.metricas(yte, c.predecir(wg, c.umbral_medio(wg, Xtr, ytr), Xte))
            filas.append(dict(rep=rep, gamma=g, balanceada=m["balanceada"], auc=c.auc(Xte @ wg, yte),
                              angulo_dif_medias=np.degrees(np.arccos(min(1, abs(wg @ ref["Dif. de medias"]))))))
        for nombre in ("Dif. de medias", "Pesos iguales", "PCA (1ª comp.)"):
            wr = ref[nombre]
            m = c.metricas(yte, c.predecir(wr, c.umbral_medio(wr, Xtr, ytr), Xte))
            filas.append(dict(rep=rep, gamma=nombre, balanceada=m["balanceada"], auc=c.auc(Xte @ wr, yte)))
    res = pd.DataFrame(filas)
    camino = res[res.gamma.apply(lambda v: not isinstance(v, str))].groupby("gamma")[
        ["balanceada", "auc", "angulo_dif_medias"]].agg(["mean", "std"])
    refs = res[res.gamma.apply(lambda v: isinstance(v, str))].groupby("gamma")[["balanceada", "auc"]].agg(["mean", "std"])
    print("\n=== Camino de regularización (PRUEBA, media y desv. de", N_REP, "particiones) ===")
    print(camino.round(3))
    print("\nReferencias:")
    print(refs.round(3))
    camino.to_csv(os.path.join(c.RESULTADOS, "camino_regularizacion.csv"))
    refs.to_csv(os.path.join(c.RESULTADOS, "camino_regularizacion_referencias.csv"))

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(GAMMAS))
    for ax, col, nombre in ((axs[0], "balanceada", "Exactitud balanceada"), (axs[1], "auc", "AUC")):
        ax.errorbar(x, camino[(col, "mean")], camino[(col, "std")], fmt="o-", capsize=3, label="Fisher con S_W + λI")
        for ref_n, sty in (("Dif. de medias", ":"), ("Pesos iguales", "--"), ("PCA (1ª comp.)", "-.")):
            ax.axhline(refs.loc[ref_n, (col, "mean")], ls=sty, lw=1.2, color="k" if ref_n != "Pesos iguales" else "tab:red",
                       alpha=.8, label=ref_n)
        ax.set_xticks(x); ax.set_xticklabels([f"{g:g}" for g in GAMMAS], rotation=45, fontsize=8)
        ax.set_xlabel("γ   (λ = γ·tr(S_W)/p)"); ax.set_ylabel(f"{nombre} (prueba)")
    axs[0].legend(fontsize=7)
    fig.suptitle("γ = 0 es Fisher; γ → ∞ lleva w a la diferencia de medias", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(c.FIGURAS, "camino_regularizacion.png"), dpi=150)
    plt.show()
    return tabla, camino


if __name__ == "__main__":
    espectro, camino = main()
