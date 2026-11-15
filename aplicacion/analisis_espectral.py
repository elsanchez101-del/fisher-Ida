# -*- coding: utf-8 -*-
"""
analisis_espectral.py
---------------------
¿Por qué Fisher no supera a la diferencia de medias con las acciones? Dos análisis.

1. DESCOMPOSICIÓN ESPECTRAL DE w.
   Sigma = S_W / (N - 2) es simétrica, así que (teorema espectral) Sigma = U diag(lambda_i) U^T,
   con autovectores u_i ortonormales. Con d = mu2 - mu1:
       w = Sigma^{-1} d = sum_i (u_i^T d / lambda_i) u_i .
   Cada autodirección u_i entra en w con peso (u_i^T d) / lambda_i: Fisher DIVIDE entre el
   autovalor. La tabla muestra cuánto pesa cada u_i en w.

2. CAMINO DE REGULARIZACIÓN.
   w_gamma = (S_W + lambda I)^{-1} d, con lambda = gamma tr(S_W)/p. Con gamma = 0 es Fisher;
   con gamma muy grande, w_gamma apunta como d (diferencia de medias). Se mide la exactitud
   balanceada en PRUEBA con las mismas 20 particiones de escenarios_lda_pca.py y el ángulo
   entre w_gamma y d.

USO: F5 en Spyder o `python aplicacion/analisis_espectral.py`.
Solo usa las 300 acciones de entrenamiento (no mira la validación externa).
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
    """Autovalores y autovectores de Sigma = S_W/(N-2) y el peso de cada autovector en w."""
    mu1, mu2, SW = c.estadisticos(X, y)
    Sigma = SW / (len(y) - 2)
    lam, U = np.linalg.eigh(Sigma)          # eigh: para matrices simétricas; orden creciente
    lam, U = lam[::-1], U[:, ::-1]          # de mayor a menor autovalor
    d = mu2 - mu1
    proy = U.T @ d                          # u_i^T d: cuánto de la diferencia de medias cae en u_i
    peso = proy / lam                       # peso de u_i en w = Sigma^{-1} d
    return lam, U, d, proy, peso, Sigma


def main():
    pd.set_option("display.width", 200)
    c.asegurar_carpetas()
    df, X, y = c.cargar()
    (Xs,) = c.estandarizar(X)

    # ---------------------------- 1) ESPECTRO ------------------------------
    lam, U, d, proy, peso, Sigma = descomposicion(Xs, y)
    w = U @ peso
    assert np.allclose(w, np.linalg.solve(Sigma, d))      # comprobación: la suma da Sigma^{-1} d
    tabla = pd.DataFrame({"lambda_i": lam, "u_i^T d": proy, "peso (u_i^T d)/lambda_i": peso,
                          "% de ||w||^2": 100 * peso ** 2 / np.sum(peso ** 2)},
                         index=[f"u{i + 1}" for i in range(len(lam))])
    autovec = pd.DataFrame(U, index=c.NOMBRES, columns=tabla.index)
    print("=== Descomposición espectral de w = Sigma^{-1} d (variables estandarizadas, 300 acciones) ===")
    print(tabla.round(3))
    print("\nAutovectores u_i (columnas):")
    print(autovec.round(3))
    print(f"\nNúmero de condición de Sigma (mayor / menor autovalor): {lam[0] / lam[-1]:.1f}")
    tabla.to_csv(os.path.join(c.RESULTADOS, "espectro_w.csv"))
    autovec.to_csv(os.path.join(c.RESULTADOS, "espectro_autovectores.csv"))

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    pos = np.arange(len(lam))
    axs[0].bar(pos, lam, color="tab:gray"); axs[0].set_title("Autovalores λᵢ de Σ")
    axs[1].bar(pos, tabla["% de ||w||^2"], color="tab:blue"); axs[1].set_title("% de ‖w‖² que aporta cada uᵢ")
    for ax in axs:
        ax.set_xticks(pos); ax.set_xticklabels(tabla.index)
    fig.suptitle("Casi todo w está en u₁, la dirección de \"riesgo general\"", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(c.FIGURAS, "espectro_w.png"), dpi=150)

    # ---------------------- 2) CAMINO DE REGULARIZACIÓN ---------------------
    rng = np.random.default_rng(c.SEMILLA)            # mismas particiones que escenarios_lda_pca.py
    filas = []
    for rep in range(N_REP):
        tr, te = c.separar(y, rng, PROP_PRUEBA)
        Xtr, Xte = c.estandarizar(X[tr], X[te])
        ytr, yte = y[tr], y[te]
        w_medias = c.orientar(c.dir_medias(Xtr, ytr), Xtr, ytr)
        for g in GAMMAS:
            wg = c.orientar(c.dir_fisher(Xtr, ytr, g), Xtr, ytr)
            m = c.metricas(yte, c.predecir(wg, c.umbral_medio(wg, Xtr, ytr), Xte))
            filas.append(dict(rep=rep, gamma=g, balanceada=m["balanceada"],
                              angulo_dif_medias=np.degrees(np.arccos(min(1, abs(wg @ w_medias))))))
        m = c.metricas(yte, c.predecir(w_medias, c.umbral_medio(w_medias, Xtr, ytr), Xte))
        filas.append(dict(rep=rep, gamma="Dif. de medias", balanceada=m["balanceada"], angulo_dif_medias=0.0))
    res = pd.DataFrame(filas)
    es_ref = res.gamma.apply(lambda v: isinstance(v, str))
    camino = res[~es_ref].groupby("gamma")[["balanceada", "angulo_dif_medias"]].agg(["mean", "std"])
    ref_medias = res[es_ref]["balanceada"].mean()
    print("\n=== Camino de regularización (PRUEBA, media y desv. de", N_REP, "particiones) ===")
    print(camino.round(3))
    print(f"\nDiferencia de medias, exactitud balanceada: {ref_medias:.3f}")
    camino.to_csv(os.path.join(c.RESULTADOS, "camino_regularizacion.csv"))

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(GAMMAS))
    axs[0].errorbar(x, camino[("balanceada", "mean")], camino[("balanceada", "std")], fmt="o-", capsize=3,
                    label="Fisher con S_W + λI")
    axs[0].axhline(ref_medias, ls=":", color="k", label="Diferencia de medias")
    axs[0].set_ylabel("Exactitud balanceada (prueba)"); axs[0].legend(fontsize=8)
    axs[1].plot(x, camino[("angulo_dif_medias", "mean")], "o-")
    axs[1].set_ylabel("Ángulo entre w y d (grados)")
    for ax in axs:
        ax.set_xticks(x); ax.set_xticklabels([f"{g:g}" for g in GAMMAS], rotation=45, fontsize=8)
        ax.set_xlabel("γ   (λ = γ·tr(S_W)/p)")
    fig.suptitle("γ = 0 es Fisher; al crecer γ, w gira hasta la diferencia de medias y la exactitud no cambia",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(c.FIGURAS, "camino_regularizacion.png"), dpi=150)
    plt.show()
    return tabla, camino


if __name__ == "__main__":
    espectro, camino = main()
