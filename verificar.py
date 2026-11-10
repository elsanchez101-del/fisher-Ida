# -*- coding: utf-8 -*-
"""
verificar.py
------------
Comprueba con un solo comando que todo el proyecto se reproduce:

  1. corre las pruebas de tests/ (pytest),
  2. vuelve a generar los experimentos sintéticos (experiments/run_all.py),
  3. vuelve a generar el caso aplicado (aplicacion/correr_todo.py, sin internet),
  4. compara cada tabla .csv y .json generada con la que está guardada en el repositorio, y al final
     deja las tablas guardadas como estaban (las figuras sí quedan regeneradas).

USO:  python verificar.py
Termina con "TODO OK" si las pruebas pasan y las tablas coinciden; si no, dice qué archivo cambió y cuánto.
Los números se comparan con tolerancia (absoluta 1e-5 y relativa 1e-6). Hay una excepción conocida:
en el experimento B con gamma = 0, S_W es singular (número de condición ~1e17) y el resultado depende
del redondeo de cada computadora (sección 7 del documento). Por eso la fila gamma = 0 de exp_reg.csv
y las celdas de resumen.json que dependen de ella (número de condición y el resultado de la
validación cruzada, que a veces elige gamma = 0) se reportan como AVISO y no como
fallo. Todo lo demás (experimento A, ejemplo a mano, comparación con sklearn y el caso aplicado
completo) debe coincidir siempre.
Las figuras (.png) no se comparan porque dependen de la versión de matplotlib.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

RAIZ = pathlib.Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):          # la consola de Windows no es UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CARPETAS_TABLAS = ["results", "aplicacion/resultados"]
EXTENSIONES = {".csv", ".json"}
TOL_ABS, TOL_REL = 1e-5, 1e-6


def valores(p):
    """Lista de celdas (texto o número) de un .csv o .json, para compararlas una a una."""
    if p.suffix == ".json":
        plano = []

        def aplanar(x):
            if isinstance(x, dict):
                for k in sorted(x):
                    plano.append(k); aplanar(x[k])
            elif isinstance(x, list):
                for v in x: aplanar(v)
            else:
                plano.append(x)
        aplanar(json.loads(p.read_text(encoding="utf-8")))
        return plano
    celdas = []
    for linea in p.read_text(encoding="utf-8").splitlines():
        for c in linea.split(","):
            try:
                celdas.append(float(c))
            except ValueError:
                celdas.append(c.strip())
    return celdas


def iguales(x, y):
    if isinstance(x, float) and isinstance(y, float):
        return abs(x - y) <= TOL_ABS + TOL_REL * max(abs(x), abs(y))
    return x == y


def comparar(a, b):
    """Devuelve (número de celdas distintas, mayor diferencia relativa)."""
    va, vb = valores(a), valores(b)
    if len(va) != len(vb):
        return max(len(va), len(vb)), float("inf")
    distintas, peor = 0, 0.0
    for x, y in zip(va, vb):
        if not iguales(x, y):
            distintas += 1
            peor = max(peor, abs(x - y) / max(abs(x), abs(y)) if isinstance(x, float) else float("inf"))
    return distintas, peor


# Celdas del experimento B que dependen del caso singular gamma = 0 (ver encabezado).
def diferencias_esperadas(nombre, a, b):
    """Devuelve (lista de avisos, True si TODAS las diferencias son las esperadas)."""
    avisos = []
    if nombre.endswith("exp_reg.csv"):
        fa = a.read_text(encoding="utf-8").splitlines(); fb = b.read_text(encoding="utf-8").splitlines()
        if len(fa) != len(fb):
            return avisos, False
        for la, lb in zip(fa, fb):
            if la != lb:
                if la.startswith("0.0,") and lb.startswith("0.0,"):
                    avisos.append(f"fila gamma = 0: guardada {la[4:]} / obtenida {lb[4:]}")
                else:
                    return avisos, False
        return avisos, True
    if nombre.endswith("resumen.json"):
        ja = json.loads(a.read_text(encoding="utf-8")); jb = json.loads(b.read_text(encoding="utf-8"))
        permitidas = {("singularidad", "cond"), ("gamma_cv",), ("acc_cv_media",), ("acc_cv_std",)}

        def recorrer(x, y, ruta):
            if isinstance(x, dict):
                if set(x) != set(y):
                    return False
                return all(recorrer(x[k], y[k], ruta + (k,)) for k in x)
            if isinstance(x, list) and ruta not in permitidas:
                return len(x) == len(y) and all(recorrer(u, v, ruta) for u, v in zip(x, y))
            if ruta in permitidas:
                if x != y:
                    avisos.append(f"{'.'.join(ruta)}: guardado {x} / obtenido {y}")
                return True
            return iguales(x, y)
        return avisos, recorrer(ja, jb, ())
    return avisos, False


def correr(descripcion, comando):
    print(f"\n=== {descripcion} ===")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    r = subprocess.run([sys.executable] + comando, cwd=RAIZ, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    print(r.stdout[-1500:] if len(r.stdout) > 1500 else r.stdout)
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise SystemExit(f"FALLÓ: {descripcion}")


def tablas(carpeta):
    return sorted(p for p in (RAIZ / carpeta).glob("*") if p.suffix in EXTENSIONES)


def main():
    # 1) Copia de seguridad de las tablas guardadas, para compararlas después.
    respaldo = pathlib.Path(tempfile.mkdtemp(prefix="fisher_tablas_"))
    guardadas = {}
    for carpeta in CARPETAS_TABLAS:
        for p in tablas(carpeta):
            destino = respaldo / carpeta / p.name
            destino.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, destino)
            guardadas[f"{carpeta}/{p.name}"] = destino

    try:
        # 2) Pruebas, experimentos sintéticos y caso aplicado.
        correr("Pruebas (tests/)", ["-m", "pytest", "-q", "tests"])
        correr("Experimentos sintéticos (experiments/run_all.py)", ["experiments/run_all.py"])
        correr("Caso aplicado (aplicacion/correr_todo.py)", ["aplicacion/correr_todo.py"])

        # 3) Comparación tabla por tabla.
        print("\n=== Comparación con las tablas guardadas en el repositorio ===")
        distintas, avisados = [], []
        for nombre, copia in guardadas.items():
            actual = RAIZ / nombre
            if not actual.exists():
                distintas.append(f"{nombre} (no se generó)")
                continue
            n, peor = comparar(copia, actual)
            if n == 0:
                print(f"  igual    {nombre}")
                continue
            avisos, esperado = diferencias_esperadas(nombre, copia, actual)
            if esperado:
                print(f"  AVISO    {nombre} (solo difiere el caso singular gamma = 0, esperado)")
                for a in avisos:
                    print(f"             {a}")
                avisados.append(nombre)
            else:
                distintas.append(f"{nombre}: {n} celda(s) distintas, diferencia relativa máxima {peor:.2e}")
        for d in distintas:
            print(f"  DISTINTA {d}")
    finally:
        # 4) Se dejan las tablas guardadas tal como estaban (las del repositorio son la referencia).
        for nombre, copia in guardadas.items():
            shutil.copy2(copia, RAIZ / nombre)
        shutil.rmtree(respaldo, ignore_errors=True)

    if distintas:
        raise SystemExit(f"\n{len(distintas)} tabla(s) no coinciden con las guardadas.")
    if avisados:
        print(f"\nTODO OK: todas las pruebas pasan y {len(guardadas) - len(avisados)} tablas coinciden; en "
              f"{', '.join(avisados)} solo cambia el caso singular gamma = 0, que depende de la computadora.")
    else:
        print(f"\nTODO OK: {len(guardadas)} tablas iguales a las guardadas y todas las pruebas pasan.")


if __name__ == "__main__":
    main()
