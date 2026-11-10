# -*- coding: utf-8 -*-
"""
comparacion_sin_umbral.py
-------------------------
Separa las dos decisiones del clasificador:
    - la DIRECCIÓN w (lo que aporta Fisher), y
    - el UMBRAL t (una decisión aparte: el punto medio supone clases de igual tamaño).
El AUC mide solo la dirección: es la probabilidad de que una acción riesgosa tenga mayor puntaje
w^T x que una no riesgosa, para cualquier umbral.

1. Validación interna: AUC en PRUEBA de cada dirección en las mismas 20 particiones.
2. Validación externa (ANÁLISIS POSTERIOR, no cambia el modelo ni el resultado principal):
   - AUC de cada dirección en las acciones nuevas;
   - intervalo de confianza de Wilson al 95 % para cada exactitud;
   - prueba exacta de McNemar entre Fisher y cada método (¿la diferencia de aciertos es ruido?).

USO: F5 en Spyder o `python aplicacion/comparacion_sin_umbral.py`.
Necesita datos/acciones_dataset_300.csv y datos/acciones_validacion.csv.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

import comun as c

N_REP       = 20
PROP_PRUEBA = 0.30
METODOS     = ["LDA (Fisher)", "PCA (1ª comp.)", "Dif. de medias", "Pesos iguales"]


def wilson(aciertos, n, conf=0.95):
    """Intervalo de Wilson para una proporción (funciona bien con n pequeño)."""
    z = norm.ppf(1 - (1 - conf) / 2)
    p = aciertos / n
    centro = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    radio = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return centro - radio, centro + radio


def mcnemar(correcto_a, correcto_b):
    """Prueba exacta de McNemar: solo cuentan las acciones donde los métodos discrepan."""
    b = int(np.sum(correcto_a & ~correcto_b))      # acierta A, falla B
    k = int(np.sum(~correcto_a & correcto_b))      # falla A, acierta B
    p = binomtest(b, b + k, 0.5).pvalue if b + k > 0 else 1.0
    return b, k, p


def main():
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    c.asegurar_carpetas()
    df, X, y = c.cargar()

    # ----------------------- 1) AUC en validación interna ------------------
    rng = np.random.default_rng(c.SEMILLA)
    filas = []
    for rep in range(N_REP):
        tr, te = c.separar(y, rng, PROP_PRUEBA)
        Xtr, Xte = c.estandarizar(X[tr], X[te])
        for m, w in c.direcciones(Xtr, y[tr]).items():
            filas.append(dict(rep=rep, metodo=m, auc=c.auc(Xte @ w, y[te])))
    interna = pd.DataFrame(filas).groupby("metodo")["auc"].agg(["mean", "std"]).reindex(METODOS)
    print(f"=== AUC en PRUEBA, validación interna ({N_REP} particiones) ===")
    print(interna.round(3))
    interna.to_csv(os.path.join(c.RESULTADOS, "auc_interna.csv"))

    # --------------- 2) Validación externa: análisis posterior --------------
    if not os.path.exists(c.ARCHIVO_NUEVAS):
        print(f"\nNo existe {c.ARCHIVO_NUEVAS}: corre antes validacion_externa.py.")
        return interna, None
    m_, s_ = X.mean(0), X.std(0, ddof=1)
    Z = (X - m_) / s_
    nv = pd.read_csv(c.ARCHIVO_NUEVAS, index_col="ticker")
    Zn = (c.a_matriz(nv) - m_) / s_
    real = nv["etiqueta"].to_numpy(dtype=int)
    n = len(real)
    dirs = c.direcciones(Z, y)
    correcto, filas = {}, []
    for k, w in dirs.items():
        pred = c.predecir(w, c.umbral_medio(w, Z, y), Zn)
        correcto[k] = pred == real
        a = int(correcto[k].sum())
        lo, hi = wilson(a, n)
        filas.append(dict(metodo=k, aciertos=a, exactitud=a / n, ic95_inf=lo, ic95_sup=hi,
                          auc=c.auc(Zn @ w, real)))
    externa = pd.DataFrame(filas).set_index("metodo")
    print(f"\n=== Validación externa ({n} acciones nuevas) — ANÁLISIS POSTERIOR ===")
    print(externa.round(3))
    filas = []
    for k in METODOS[1:]:
        b, kk, p = mcnemar(correcto["LDA (Fisher)"], correcto[k])
        filas.append(dict(comparacion=f"Fisher vs {k}", solo_acierta_fisher=b, solo_acierta_otro=kk, p_valor=p))
    mc = pd.DataFrame(filas).set_index("comparacion")
    print("\nMcNemar exacto (p > 0.05: la diferencia de aciertos no es distinguible del azar):")
    print(mc.round(3))
    externa.to_csv(os.path.join(c.RESULTADOS, "externa_auc_ic.csv"))
    mc.to_csv(os.path.join(c.RESULTADOS, "externa_mcnemar.csv"))
    return interna, externa


if __name__ == "__main__":
    auc_interna, externa = main()
