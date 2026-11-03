# Discriminante lineal de Fisher (LDA) — avance del proyecto

Álgebra Lineal Aplicada · Universidad del Pacífico. Proyecto grupal.
Integrantes: _(completar)_

Implementación **desde cero** (numpy) del discriminante de Fisher para dos clases (y extensión multiclase),
con verificación contra `scikit-learn` y experimentos reproducibles.

## Estructura
```
src/fisher.py            implementación (medias, S_W, dirección w, criterio J, clasificador, multiclase, regularización)
tests/test_fisher.py     pruebas: ejemplo a mano, invarianzas, máximo de J, comparación con sklearn, rango de S_B, S_W singular
experiments/run_all.py   todos los experimentos (semilla fija 2026)
figures/  results/       salidas generadas (png, csv, json)
docs/                    desarrollo matemático (PDF) y guion del video
```

## Uso
```bash
pip install -r requirements.txt
python -m pytest -q tests          # 8 pruebas
python experiments/run_all.py      # regenera figuras y tablas
```

## Qué se calcula
- `w` resuelve `(S_W + λI) w = μ1 − μ2` con `np.linalg.solve` (no se invierte la matriz); `λ = γ·tr(S_W)/p`.
- Umbral: punto medio de las medias proyectadas **de entrenamiento** (priors iguales, costos simétricos).
- Comparación: diferencia de medias, Fisher y PC1 (PCA calculado solo con entrenamiento), misma regla de umbral.
- Separación train/test antes del ajuste; γ elegido por validación cruzada **dentro** del entrenamiento.

## Reproducibilidad
Semillas `2026+k` (experimento gaussiano) y `2126+k` (regularización), k = 0..9.

## Limitaciones
Datos sintéticos gaussianos con covarianza común (favorables a LDA). Fisher no domina en general: covarianzas distintas,
clases no linealmente separables o etiquetas ruidosas pueden cambiar el resultado.
