# -*- coding: utf-8 -*-
"""
verificar.py
------------
Comprueba con un solo comando que el proyecto se reproduce.

  1. Copia el proyecto a una carpeta temporal (así no se toca nada del original).
  2. En la copia: corre las pruebas, los experimentos sintéticos y el caso aplicado.
  3. Compara cada tabla .csv que se generó con la que está guardada en el repositorio.

USO:  python verificar.py
Termina con "TODO OK" si las pruebas pasan y las tablas coinciden.

Las computadoras pueden diferir en el último decimal, así que dos números se consideran iguales
si su diferencia es menor que 0.00001. Única excepción: en el experimento B con gamma = 0 la matriz
S_W es singular y ese resultado es inestable (puede cambiar un poco entre computadoras); si esa
fila cambia, se muestra un aviso y no un error.
"""
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

RAIZ = pathlib.Path(__file__).resolve().parent
CARPETAS = ["results", "aplicacion/resultados"]
TOLERANCIA = 1e-5


def correr(titulo, comando, carpeta):
    print(f"\n=== {titulo} ===")
    r = subprocess.run([sys.executable] + comando, cwd=carpeta, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env={**os.environ, "PYTHONUTF8": "1"})
    print(r.stdout[-800:])
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise SystemExit(f"FALLÓ: {titulo}")


def tablas_iguales(a, b):
    """True si las dos tablas tienen la misma forma y los mismos números (con tolerancia)."""
    A, B = pd.read_csv(a), pd.read_csv(b)
    if A.shape != B.shape or list(A.columns) != list(B.columns):
        return False
    numA = A.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    numB = B.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return bool(np.allclose(numA, numB, atol=TOLERANCIA, rtol=0, equal_nan=True))


def main():
    copia = pathlib.Path(tempfile.mkdtemp(prefix="fisher_")) / "proyecto"
    shutil.copytree(RAIZ, copia, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    try:
        correr("Pruebas", ["-m", "pytest", "-q", "tests"], copia)
        correr("Experimentos sintéticos", ["experiments/run_all.py"], copia)
        correr("Caso aplicado", ["aplicacion/correr_todo.py"], copia)

        print("\n=== Comparación con las tablas guardadas ===")
        errores = 0
        for carpeta in CARPETAS:
            for guardada in sorted((RAIZ / carpeta).glob("*.csv")):
                nueva = copia / carpeta / guardada.name
                nombre = f"{carpeta}/{guardada.name}"
                if nueva.exists() and tablas_iguales(guardada, nueva):
                    print(f"  igual   {nombre}")
                elif (guardada.name == "exp_reg.csv" and nueva.exists()
                      and tablas_iguales(_sin_gamma0(guardada), _sin_gamma0(nueva))):
                    print(f"  aviso   {nombre}: solo cambia la fila gamma = 0 (S_W singular, resultado inestable)")
                else:
                    print(f"  DISTINTA {nombre}")
                    errores += 1
    finally:
        shutil.rmtree(copia.parent, ignore_errors=True)

    if errores:
        raise SystemExit(f"\n{errores} tabla(s) no coinciden con las guardadas.")
    print("\nTODO OK: todas las pruebas pasan y las tablas coinciden con las guardadas.")


def _sin_gamma0(ruta):
    """Escribe una copia de exp_reg.csv sin la fila gamma = 0 y devuelve su ruta."""
    salida = pathlib.Path(tempfile.mkstemp(suffix=".csv")[1])
    pd.read_csv(ruta).query("gamma != 0").to_csv(salida, index=False)
    return salida


if __name__ == "__main__":
    main()
