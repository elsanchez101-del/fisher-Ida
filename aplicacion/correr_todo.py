# -*- coding: utf-8 -*-
"""
correr_todo.py
--------------
Corre todos los análisis del caso aplicado, en orden, a partir de los datos guardados en datos/.
No descarga nada: si falta un archivo, avisa y se detiene.

USO: F5 en Spyder o `python aplicacion/correr_todo.py`.
"""
import os
import sys
import warnings
import matplotlib
matplotlib.use("Agg")          # guarda las figuras sin abrir ventanas
warnings.filterwarnings("ignore", message="FigureCanvasAgg")   # plt.show() no hace nada con Agg
if hasattr(sys.stdout, "reconfigure"):                           # en Windows la consola no es UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import comun as c

for archivo in (c.ARCHIVO, c.ARCHIVO_NUEVAS):
    if not os.path.exists(archivo):
        raise SystemExit(f"Falta {archivo}. Los datos se generan con datos_300.py y validacion_externa.py.")

import escenarios_lda_pca, validacion_externa, analisis_espectral, comparacion_sin_umbral, \
    estabilidad_pesos, tres_clases

for modulo in (escenarios_lda_pca, validacion_externa, analisis_espectral, comparacion_sin_umbral,
               estabilidad_pesos, tres_clases):
    print("\n" + "=" * 100 + f"\n{modulo.__name__}.py\n" + "=" * 100)
    modulo.main()

print(f"\nListo. Tablas en {c.RESULTADOS} y figuras en {c.FIGURAS}.")
