# -*- coding: utf-8 -*-
"""
validacion_externa.py
---------------------
PRUEBA FINAL del modelo con acciones NUEVAS (no usadas en el entrenamiento ni en los experimentos).

Pasos:
  1. Entrena con las 300 acciones de datos/acciones_dataset_300.csv (estandariza con ellas).
  2. Acciones nuevas: si existe datos/acciones_validacion.csv se usa ese archivo (así la prueba es
     siempre la misma y se puede reproducir). Si no existe, se descargan N_NUEVAS acciones del
     S&P 500 que datos_300.py no procesó (siguientes en la cola) y se guardan en ese archivo.
  3. El modelo predice 1 o 2 para cada acción nueva SIN ver la etiqueta real.
  4. Se compara lo predicho con lo que realmente pasó (mdd_futuro > UMBRAL => riesgosa).

Métodos comparados (misma regla de umbral en los que usan dirección w):
  LDA (Fisher) | PCA (1ª comp.) | Dif. de medias | Indicador de pesos iguales | Regla de persistencia
  (persistencia: riesgosa si la caída máxima PASADA ya superó UMBRAL; no usa entrenamiento)

IMPORTANTE: esta prueba se usa UNA sola vez, al final. No se cambia el modelo mirando estos resultados.
USO: F5 en Spyder o `python aplicacion/validacion_externa.py`.
"""
import os
import numpy as np
import pandas as pd

import comun as c

# ----------------------------- PARÁMETROS ---------------------------------
N_NUEVAS = 40            # solo se usa si hay que descargar (si el CSV ya existe, se usa tal cual)
GAMMA    = 0.0           # regularización de LDA: S_W + lambda*I, lambda = GAMMA*tr(S_W)/p
METODOS  = ["LDA (Fisher)", "PCA (1ª comp.)", "Dif. de medias", "Pesos iguales", "Regla persistencia"]
SALIDA   = os.path.join(c.RESULTADOS, "validacion_externa.csv")


def acciones_nuevas():
    """Lee las acciones de la validación externa; solo las descarga si aún no existen."""
    if os.path.exists(c.ARCHIVO_NUEVAS):
        print(f"Usando las acciones guardadas en {c.ARCHIVO_NUEVAS} (no se descarga nada).")
        return pd.read_csv(c.ARCHIVO_NUEVAS, index_col="ticker")
    import datos_300 as dd
    usados = int(open(c.ARCHIVO_COLA).read())          # tickers que datos_300.py ya consumió
    cola = dd.cola_de_acciones()[usados:]
    nuevas, _ = dd.descargar_cola(cola, N_NUEVAS, etiqueta_lote="Lote nuevo")
    if len(nuevas) < N_NUEVAS:
        print(f"AVISO: solo hay {len(nuevas)} acciones nuevas disponibles (se pedían {N_NUEVAS}).")
    nuevas.to_csv(c.ARCHIVO_NUEVAS)
    print(f"Acciones nuevas guardadas en {c.ARCHIVO_NUEVAS}")
    return nuevas


def main():
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    c.asegurar_carpetas()
    # 1) ENTRENAR con las 300 (la estandarización usa solo estos datos)
    tr, Xtr, ytr = c.cargar()
    m, s = Xtr.mean(0), Xtr.std(0, ddof=1)
    Ztr = (Xtr - m) / s
    dirs = c.direcciones(Ztr, ytr, GAMMA)
    umbral = {k: c.umbral_medio(w, Ztr, ytr) for k, w in dirs.items()}
    print(f"Modelo entrenado con {len(ytr)} acciones ({np.sum(ytr == 1)} no riesgosas, {np.sum(ytr == 2)} riesgosas)")
    print("Pesos w (variables estandarizadas):")
    print(pd.DataFrame(dirs, index=c.NOMBRES).round(3))

    # 2) ACCIONES NUEVAS y su resultado real
    nv = acciones_nuevas()
    Zn = (c.a_matriz(nv) - m) / s                    # misma estandarización que en el entrenamiento
    real = nv["etiqueta"].to_numpy(dtype=int)

    # 3) PREDECIR (sin usar la etiqueta real)
    pred = {k: c.predecir(w, umbral[k], Zn) for k, w in dirs.items()}
    pred[METODOS[4]] = np.where(nv["mdd"].to_numpy() > c.UMBRAL, 2, 1)

    # 4) COMPARAR con lo que realmente pasó
    N = len(real)
    filas = {k: {kk: vv for kk, vv in c.metricas(real, p).items() if kk != "cm"} for k, p in pred.items()}
    tabla = pd.DataFrame(filas).T
    tabla.to_csv(os.path.join(c.RESULTADOS, "validacion_externa_resumen.csv"))
    print(f"\n=== Validación externa: {N} acciones nuevas ===")
    print(f"Reales: {np.sum(real == 1)} no riesgosas y {np.sum(real == 2)} riesgosas")
    print(f"Línea base (decir siempre la clase mayoritaria): {max(np.mean(real == 1), np.mean(real == 2)):.3f}")
    print(tabla.round(3))
    for k, p in pred.items():
        print(f"\nMatriz de confusión - {k} (filas = real 1,2; columnas = predicho 1,2):")
        print(c.metricas(real, p)["cm"])

    # 5) Detalle por acción
    det = nv[c.VARS + ["mdd_futuro"]].copy()
    det["real"] = real
    for k, p in pred.items():
        det[k] = p
    det.to_csv(SALIDA)
    print(f"\nDetalle por acción guardado en {SALIDA}")
    fn = det[(det[METODOS[0]] == 1) & (det["real"] == 2)]
    fp = det[(det[METODOS[0]] == 2) & (det["real"] == 1)]
    print(f"\nLDA - FALSOS NEGATIVOS (dijo 'no riesgosa' pero cayó > {c.UMBRAL:.0%}): {len(fn)}")
    print(fn[c.VARS + ["mdd_futuro"]].round(3))
    print(f"\nLDA - FALSOS POSITIVOS (dijo 'riesgosa' pero no cayó tanto): {len(fp)}")
    print(fp[c.VARS + ["mdd_futuro"]].round(3))
    return det, tabla


if __name__ == "__main__":
    detalle, resultados = main()
