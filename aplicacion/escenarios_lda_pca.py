# -*- coding: utf-8 -*-
"""
escenarios_lda_pca.py
---------------------
Validación INTERNA: dos escenarios con los MISMOS datos y la MISMA regla de clasificación;
solo cambia la dirección w:
    Escenario A: dirección de Fisher (LDA)        -> usa las etiquetas
    Escenario B: dirección de PCA (1ª componente) -> NO usa las etiquetas
(Se añade la diferencia de medias como referencia.)

Regla común: puntaje = w^T x (x estandarizado); umbral = punto medio de las medias proyectadas
de las dos clases (calculadas en ENTRENAMIENTO); puntaje > umbral => riesgosa (2).
Orientación: se elige el signo de w para que las riesgosas tengan puntaje mayor (el signo no
cambia la separación; solo hace que "puntaje alto = más riesgo").

Se repite N_REP veces con separaciones entrenamiento/prueba distintas (estratificadas).

USO: F5 en Spyder o `python aplicacion/escenarios_lda_pca.py`.
Lee datos/acciones_dataset_300.csv. Escribe en resultados/ y figuras/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import comun as c

# ----------------------------- PARÁMETROS ---------------------------------
N_REP       = 20        # número de separaciones entrenamiento/prueba
PROP_PRUEBA = 0.30
METODOS     = ["LDA (Fisher)", "PCA (1ª comp.)", "Dif. de medias"]


def evaluar(w, Xtr, ytr, Xte, yte):
    t = c.umbral_medio(w, Xtr, ytr)
    pred = c.predecir(w, t, Xte)
    m = c.metricas(yte, pred)
    return dict(exactitud=m["exactitud"], balanceada=m["balanceada"],
                recall_riesgosa=m["riesgosas_detectadas"], cm=m["cm"], t=t,
                pred1=m["etiq_como_1"], pred2=m["etiq_como_2"])


def main():
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    c.asegurar_carpetas()
    df, X, y = c.cargar()
    print(f"{len(y)} acciones: {np.sum(y == 1)} no riesgosas, {np.sum(y == 2)} riesgosas | "
          f"log(1+D/E): {c.USAR_LOG_DE}")
    rng = np.random.default_rng(c.SEMILLA)
    filas, cms, angulos, primero = [], {m: np.zeros((2, 2), int) for m in METODOS}, [], None

    for rep in range(N_REP):
        tr, te = c.separar(y, rng, PROP_PRUEBA)
        Xtr, Xte = c.estandarizar(X[tr], X[te])
        ytr, yte = y[tr], y[te]
        todas = c.direcciones(Xtr, ytr)
        dirs = {m: todas[m] for m in METODOS}
        base = max(np.mean(yte == 1), np.mean(yte == 2))        # acierto de "siempre la clase mayoritaria"
        for m, w in dirs.items():
            r = evaluar(w, Xtr, ytr, Xte, yte)
            cms[m] += r["cm"]
            filas.append(dict(rep=rep, metodo=m, exactitud=r["exactitud"], balanceada=r["balanceada"],
                              recall_riesgosa=r["recall_riesgosa"], J_train=c.criterio_J(w, Xtr, ytr),
                              linea_base=base, pred1=r["pred1"], pred2=r["pred2"],
                              real1=int(np.sum(yte == 1)), real2=int(np.sum(yte == 2))))
        angulos.append({m: np.degrees(np.arccos(min(1, abs(dirs[m] @ dirs[METODOS[0]]))))
                        for m in METODOS[1:]})
        if rep == 0:
            primero = (dirs, Xtr, ytr, Xte, yte)

    res = pd.DataFrame(filas)
    res.to_csv(os.path.join(c.RESULTADOS, "escenarios_repeticiones.csv"), index=False)
    resumen = res.groupby("metodo")[["exactitud", "balanceada", "recall_riesgosa", "J_train"]].agg(["mean", "std"]).round(3)
    resumen.reindex(METODOS).to_csv(os.path.join(c.RESULTADOS, "escenarios_resumen.csv"))
    print("\n=== Resultados en PRUEBA (media y desv. de", N_REP, "repeticiones) ===")
    print(resumen.reindex(METODOS))
    print("Línea base (siempre la clase mayoritaria), exactitud media:", round(res["linea_base"].mean(), 3))
    print("\n=== Acciones etiquetadas por clase (promedio por repetición, en PRUEBA) ===")
    cuenta = res.groupby("metodo")[["pred1", "pred2"]].mean().reindex(METODOS).round(1)
    cuenta.columns = ["etiquetadas como 1 (no riesgosa)", "etiquetadas como 2 (riesgosa)"]
    print(cuenta)
    print("Reales en la prueba: ", round(res["real1"].mean(), 1), "de clase 1 y", round(res["real2"].mean(), 1), "de clase 2")
    print("\nÁngulo medio entre la dirección de LDA y las otras (grados):")
    print(pd.DataFrame(angulos).mean().round(1))
    print("\nMatriz de confusión acumulada en las", N_REP, "repeticiones (filas = real 1,2; columnas = predicho 1,2):")
    for m in METODOS:
        print(m, cms[m].tolist())

    # --- Pesos de cada variable (ajuste con TODOS los datos, solo para interpretar) ---
    (Xs,) = c.estandarizar(X)
    todas = c.direcciones(Xs, y)
    pesos = pd.DataFrame({m: todas[m] for m in METODOS}, index=c.NOMBRES).round(3)
    pesos.to_csv(os.path.join(c.RESULTADOS, "pesos.csv"))
    print("\nPesos w de cada variable (variables estandarizadas; signo + = más riesgo):")
    print(pesos)

    # --- Verificación contra scikit-learn ---
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SK
        dirs, Xtr, ytr, Xte, yte = primero
        wsk = SK(solver="lsqr").fit(Xtr, ytr).coef_[0]
        wsk = wsk / np.linalg.norm(wsk)
        print("\nVerificación con scikit-learn: |coseno| entre nuestra w y la de sklearn =",
              round(abs(dirs[METODOS[0]] @ wsk), 6), "(debe ser 1)")
    except Exception as e:
        print("\n(No se pudo verificar con scikit-learn:", e, ")")

    # --- Figuras ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([res[res.metodo == m]["balanceada"] for m in METODOS])
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(METODOS)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_ylabel("Exactitud balanceada (prueba)")
    ax.set_title(f"{N_REP} repeticiones")
    fig.tight_layout(); fig.savefig(os.path.join(c.FIGURAS, "comparacion_metodos.png"), dpi=150)

    dirs, Xtr, ytr, Xte, yte = primero
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for ax, m in zip(axs, METODOS[:2]):
        w = dirs[m]
        z = Xte @ w
        ax.hist(z[yte == 1], 10, alpha=.6, label="No riesgosa (1)")
        ax.hist(z[yte == 2], 10, alpha=.6, label="Riesgosa (2)")
        ax.axvline(c.umbral_medio(w, Xtr, ytr), color="k", ls="--", label="Umbral")
        ax.set_title(f"{m}: puntajes en prueba (rep. 1)"); ax.set_xlabel("puntaje w·x")
    axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(c.FIGURAS, "histogramas_lda_pca.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    pesos.plot(kind="bar", ax=ax); ax.axhline(0, color="k", lw=.8)
    ax.set_ylabel("peso en w"); ax.set_title("Qué pesa cada variable en la dirección")
    fig.tight_layout(); fig.savefig(os.path.join(c.FIGURAS, "pesos.png"), dpi=150)

    # --- Figura clave: puntos + dirección w (ajuste con TODOS los datos, solo ilustrativo) ---
    col = {1: "tab:blue", 2: "tab:red"}
    lab = {1: "No riesgosa (1)", 2: "Riesgosa (2)"}
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2))

    # Panel A: plano sigma-beta. Aquí SÍ se ve la recta w (modelo de 2 variables, solo para visualizar)
    ax = axs[0]
    P = Xs[:, :2]
    wA = {"LDA (Fisher)": c.orientar(c.dir_fisher(P, y), P, y),
          "PCA (1ª comp.)": c.orientar(c.dir_pca(P), P, y)}
    estilo = {"LDA (Fisher)": ("k", "-"), "PCA (1ª comp.)": ("tab:green", "--")}
    for k in (1, 2):
        ax.scatter(P[y == k, 0], P[y == k, 1], c=col[k], alpha=.65, s=28, label=lab[k])
    t = np.linspace(-4, 4, 2)
    for m, w in wA.items():
        c_, ls = estilo[m]
        ax.plot(t * w[0], t * w[1], c=c_, ls=ls, lw=2, label=f"w {m}")
        u = c.umbral_medio(w, P, y)                        # frontera: w·x = umbral (perpendicular a w)
        n = np.array([-w[1], w[0]])
        ax.plot(u * w[0] + t * n[0], u * w[1] + t * n[1], c=c_, ls=ls, lw=1, alpha=.6)
    ax.set_xlabel("sigma (estandarizada)"); ax.set_ylabel("beta (estandarizada)")
    ax.set_xlim(P[:, 0].min() - .5, P[:, 0].max() + .5); ax.set_ylim(P[:, 1].min() - .5, P[:, 1].max() + .5)
    ax.set_aspect("equal"); ax.legend(fontsize=8)
    ax.set_title("A) Plano sigma-beta: w (recta gruesa) y frontera (recta fina)")

    # Panel B: las 4 variables en el plano generado por w_LDA y la parte de w_PCA ortogonal (Gram-Schmidt)
    ax = axs[1]
    w1 = todas["LDA (Fisher)"]
    wp = todas["PCA (1ª comp.)"]
    q2 = wp - (wp @ w1) * w1
    q2 = q2 / np.linalg.norm(q2)
    a, b = Xs @ w1, Xs @ q2
    for k in (1, 2):
        ax.scatter(a[y == k], b[y == k], c=col[k], alpha=.65, s=28, label=lab[k])
    ax.axvline(c.umbral_medio(w1, Xs, y), color="k", lw=1.5, label="Umbral de Fisher")
    ax.axhline(0, color="gray", lw=.6)
    ax.set_xlabel("proyección sobre w de Fisher (las 4 variables)")
    ax.set_ylabel("dirección ortogonal (parte de w de PCA)")
    ax.legend(fontsize=8)
    ax.set_title("B) 4 variables vistas en el plano {w_LDA, w_PCA⊥}")
    fig.tight_layout(); fig.savefig(os.path.join(c.FIGURAS, "puntos_y_direccion_w.png"), dpi=150)
    plt.show()
    return res


if __name__ == "__main__":
    resultados = main()
