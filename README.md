# Discriminante lineal de Fisher (LDA)

Proyecto grupal del curso Álgebra Lineal Aplicada 2026-2, Universidad del Pacífico.

- **Tema:** Separación de clases con el discriminante de Fisher
- **Integrantes:** Elison Sanchez, Rodolfo Brandariz, Marcelo Bravo y Marcelo Torres
- **Versión actual:** entrega parcial (etiqueta `avance-parcial`)
- **Video:** se agregará en la entrega final

## De qué trata

Tenemos datos de dos clases conocidas, cada uno con varias variables, y queremos proyectarlos sobre una
sola recta, $y = w^\top x$, sin que las clases se mezclen. Proyectar sobre la recta que une las medias
falla cuando las variables están correlacionadas. Fisher elige $w$ maximizando la separación de las medias
dividida entre la dispersión de cada clase, y la solución es $w \propto S_W^{-1}(\mu_1 - \mu_2)$.

Implementamos el método desde cero con NumPy, lo comparamos con scikit-learn y lo probamos en
experimentos con datos sintéticos.

**Objetivo.** Comprobar que nuestra implementación:

1. reproduce el ejemplo a mano ($w \propto (4,1)$, umbral 15, $J = 7$);
2. da la misma dirección que `LinearDiscriminantAnalysis` de scikit-learn;
3. tiene mayor exactitud en prueba que la diferencia de medias y que PCA con datos correlacionados;
4. mejora con regularización ($S_W + \lambda I$) cuando $S_W$ es singular.

El objetivo y el alcance completos están en la sección 1 del documento
[`docs/desarrollo_matematico.pdf`](docs/desarrollo_matematico.pdf).

**Conexión con el curso.** Usamos producto interno (la proyección $y = w^\top x$), formas cuadráticas y matrices
simétricas semidefinidas positivas, la desigualdad de Cauchy–Schwarz, sistemas lineales, autovalores generalizados,
rango y condicionamiento de matrices, y regularización.

## Estado de la entrega parcial

| Lo que pide la entrega | Dónde está | Estado |
|---|---|---|
| Problema, objetivo y alcance | `docs/desarrollo.md` (sec. 1) | Listo |
| Desarrollo matemático | `docs/desarrollo.md` (sec. 2 a 6 y 9) y `docs/desarrollo_matematico.pdf` | Listo |
| Implementación | `src/fisher.py` y `experiments/run_all.py` | Listo |
| Resultados y comprobaciones | `tests/test_fisher.py`, `results/`, `figures/` y `docs/desarrollo.md` (sec. 7) | Listo |
| Fuentes citadas (IEEE) | `docs/desarrollo.md` (Referencias) | Listo |
| Guion del video | `docs/guion_video.md` y `docs/guion_video.pdf` | Listo |
| Contribuciones de cada integrante | `docs/desarrollo.md` (sec. 10) | Listo |
| Pendientes y cronograma | `docs/desarrollo.md` (sec. 8 y 11) | Listo |

## Resultados principales

**Ejemplo a mano.** El código reproduce los valores calculados a mano: $w \propto (4, 1)$, proyecciones
{5, 9, 10} y {19, 23, 24}, umbral 15 y $J = 7$.

**Datos correlacionados** (10 repeticiones, 150 puntos por clase, mitad entrenamiento y mitad prueba):

| Dirección | Exactitud en prueba |
|---|---|
| Diferencia de medias | 0.579 ± 0.034 |
| **Fisher** | **0.695 ± 0.032** |
| PCA (primera componente) | 0.559 ± 0.038 |

Fisher fue mejor en las 10 repeticiones, y su dirección coincide con la de scikit-learn (diferencia del orden de $10^{-16}$).

**Pocos datos** (30 datos y 40 variables, $S_W$ singular):

| Método | Exactitud en prueba |
|---|---|
| Sin regularizar ($\gamma = 0$) | 0.513 ± 0.035 |
| Pseudoinversa | 0.562 ± 0.027 |
| Regularizado, $\gamma$ elegido por validación cruzada | 0.619 ± 0.020 |
| Regularizado, mejor $\gamma$ de la rejilla ($\gamma = 10$) | 0.630 ± 0.010 |
| Diferencia de medias | 0.628 ± 0.010 |

Con $\gamma$ grande, $w$ tiende a la diferencia de medias, como predice la teoría.

## Organización de archivos

```
src/fisher.py            implementación: medias, S_W, dirección w, criterio J, clasificador, multiclase y regularización
tests/test_fisher.py     13 pruebas: ejemplo a mano, invarianzas, máximo de J, comparación con sklearn, rango de S_B,
                         S_W singular, medias iguales, gamma grande e intercambio de etiquetas
experiments/run_all.py   genera todos los experimentos, tablas y figuras
results/                 tablas (csv) y resumen de resultados (json)
figures/                 gráficos usados en el documento y el video
docs/                    desarrollo matemático y guion del video: el .pdf es para leer y el .md es su archivo fuente
```

## Cómo ejecutarlo

Se necesita Python 3 (lo probamos con Python 3.13).

```bash
pip install -r requirements.txt
python -m pytest -q tests          # corre las 13 pruebas
python experiments/run_all.py      # vuelve a generar results/ y figures/
```

Al ejecutar `run_all.py` se obtienen los mismos valores que están en `results/`, porque las semillas están fijas.

## Datos

No usamos archivos de datos externos. Todos los datos son sintéticos y se generan dentro de
`experiments/run_all.py`:

- **Experimento A:** dos clases gaussianas con medias $(0,0)$ y $(1,0)$ y covarianza
  $\Sigma$ con varianzas 4 y 1 y covarianza 1.8 (en Python: `[[4, 1.8], [1.8, 1]]`). Semillas $2026 + k$, con $k = 0, \dots, 9$.
- **Experimento B:** 40 variables (2 informativas, 1 copia con ruido de la primera y 37 de ruido),
  15 datos de entrenamiento por clase. Semillas $2126 + k$, con $k = 0, \dots, 9$.

## Detalles del método

- $w$ se obtiene resolviendo $(S_W + \lambda I) w = \mu_1 - \mu_2$ con `np.linalg.solve`, sin calcular la inversa, con $\lambda = \gamma \cdot \mathrm{tr}(S_W)/p$.
- El umbral es el punto medio de las medias proyectadas del entrenamiento (probabilidades previas iguales y costos simétricos).
- Las tres direcciones se comparan con la misma regla de umbral, y PCA se calcula solo con el entrenamiento.
- Entrenamiento y prueba se separan antes de ajustar, y $\gamma$ se elige con validación cruzada dentro del entrenamiento.

## Limitaciones

Los datos son sintéticos y tienen la misma covarianza en las dos clases, que es el caso favorable para
Fisher. Con covarianzas distintas, clases que no se separan con una recta o etiquetas con errores, el
resultado puede cambiar, así que no concluimos que Fisher sea siempre el mejor método.

## Referencias

Las referencias completas, en formato IEEE, están al final de
[`docs/desarrollo_matematico.pdf`](docs/desarrollo_matematico.pdf).
