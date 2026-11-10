# -*- coding: utf-8 -*-
"""
estabilidad_pesos.py
--------------------
¿Los pesos de w son un rasgo real de los datos o ruido de la muestra? (por ejemplo, el peso
negativo de la caída máxima pasada en la dirección de Fisher).

Bootstrap: se remuestrean con reemplazo las 300 acciones (dentro de cada clase, para mantener
sus tamaños) B veces; en cada muestra se estandariza y se recalcula cada dirección w. Se reporta
la media de cada peso, su intervalo del 2.5 % al 97.5 %, la proporción de veces que es negativo
y el ángulo entre la w de cada muestra y la w con todos los datos.

Lectura: si el intervalo de un peso cruza el cero, su signo no es confiable con 300 acciones.
Un ángulo grande indica que la dirección cambia mucho de una muestra a otra (estimación ruidosa).

USO: F5 en Spyder o `python aplicacion/estabilidad_pesos.py`.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import comun as c

B       = 1000
METODOS = ["LDA (Fisher)", "PCA (1ª comp.)", "Dif. de medias"]


def main():
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    c.asegurar_carpetas()
    df, X, y = c.cargar()
    (Xs,) = c.estandarizar(X)
    w_total = {m: v for m, v in c.direcciones(Xs, y).items() if m in METODOS}

    rng = np.random.default_rng(c.SEMILLA)
    idx1, idx2 = np.where(y == 1)[0], np.where(y == 2)[0]
    pesos = {m: np.empty((B, X.shape[1])) for m in METODOS}
    angulos = {m: np.empty(B) for m in METODOS}
    for b in range(B):
        idx = np.r_[rng.choice(idx1, len(idx1)), rng.choice(idx2, len(idx2))]
        (Xb,) = c.estandarizar(X[idx])
        dirs = c.direcciones(Xb, y[idx])
        for m in METODOS:
            pesos[m][b] = dirs[m]
            angulos[m][b] = np.degrees(np.arccos(min(1, abs(dirs[m] @ w_total[m]))))

    filas = []
    for m in METODOS:
        for j, v in enumerate(c.NOMBRES):
            p = pesos[m][:, j]
            lo, hi = np.percentile(p, [2.5, 97.5])
            filas.append(dict(metodo=m, variable=v, peso_total=w_total[m][j], media=p.mean(),
                              ic_inf=lo, ic_sup=hi, prop_negativo=np.mean(p < 0),
                              signo_confiable=bool(lo > 0 or hi < 0)))
    tabla = pd.DataFrame(filas)
    ang = pd.DataFrame({m: [np.median(angulos[m]), np.percentile(angulos[m], 95)] for m in METODOS},
                       index=["ángulo mediano (°)", "ángulo percentil 95 (°)"])
    print(f"=== Bootstrap de los pesos ({B} remuestras de las {len(y)} acciones) ===")
    print(tabla.round(3).to_string(index=False))
    print("\nCuánto cambia la dirección entre remuestras (ángulo con la w de todos los datos):")
    print(ang.round(1))
    tabla.to_csv(os.path.join(c.RESULTADOS, "bootstrap_pesos.csv"), index=False)
    ang.to_csv(os.path.join(c.RESULTADOS, "bootstrap_angulos.csv"))

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axs[0]
    ancho = 0.26
    for k, m in enumerate(METODOS):
        t = tabla[tabla.metodo == m]
        pos = np.arange(len(c.NOMBRES)) + (k - 1) * ancho
        ax.errorbar(pos, t.peso_total, [t.peso_total - t.ic_inf, t.ic_sup - t.peso_total],
                    fmt="o", capsize=3, label=m)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(np.arange(len(c.NOMBRES))); ax.set_xticklabels(c.NOMBRES)
    ax.set_ylabel("peso en w (intervalo 95 %)"); ax.legend(fontsize=8)
    ax.set_title("Pesos y su variación por bootstrap")
    ax = axs[1]
    ax.boxplot([angulos[m] for m in METODOS]); ax.set_xticks([1, 2, 3]); ax.set_xticklabels(METODOS, fontsize=8)
    ax.set_ylabel("ángulo con la w total (°)"); ax.set_title("Estabilidad de la dirección")
    fig.tight_layout(); fig.savefig(os.path.join(c.FIGURAS, "bootstrap_pesos.png"), dpi=150)
    plt.show()
    return tabla, ang


if __name__ == "__main__":
    tabla, angulos = main()
