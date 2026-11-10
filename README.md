# Discriminante lineal de Fisher (LDA)

Proyecto grupal del curso Álgebra Lineal Aplicada 2026-2, Universidad del Pacífico.

- **Tema:** Separación de clases con el discriminante de Fisher
- **Integrantes:** Elison Sanchez, Rodolfo Brandariz, Marcelo Bravo y Marcelo Torres
- **Versión actual:** entrega final (etiqueta `entrega-final`); la entrega parcial quedó en la etiqueta `avance-parcial`
- **Video:** (enlace pendiente de grabar)

## De qué trata

Tenemos datos de dos clases conocidas, cada uno con varias variables, y queremos proyectarlos sobre una
sola recta, $y = w^\top x$, sin que las clases se mezclen. Proyectar sobre la recta que une las medias
falla cuando las variables están correlacionadas. Fisher elige $w$ maximizando la separación de las medias
dividida entre la dispersión de cada clase, y la solución es $w \propto S_W^{-1}(\mu_1 - \mu_2)$.

Implementamos el método desde cero con NumPy, lo comparamos con scikit-learn, lo probamos en
experimentos con datos sintéticos y, para la entrega final, lo aplicamos a datos reales: riesgo de acciones
del S&P 500 (carpeta `aplicacion/`).

**Objetivo.** Comprobar que nuestra implementación:

1. reproduce el ejemplo a mano ($w \propto (4,1)$, umbral 15, $J = 7$);
2. da la misma dirección que `LinearDiscriminantAnalysis` de scikit-learn;
3. tiene mayor exactitud en prueba que la diferencia de medias y que PCA con datos correlacionados;
4. mejora con regularización ($S_W + \lambda I$) cuando $S_W$ es singular;
5. y, en datos reales, entender por qué Fisher no supera a direcciones más simples (descomposición espectral de $w$).

El objetivo y el alcance completos están en la sección 1 del documento
[`docs/desarrollo_matematico.pdf`](docs/desarrollo_matematico.pdf).

**Conexión con el curso.** Usamos producto interno (la proyección $y = w^\top x$), formas cuadráticas y matrices
simétricas semidefinidas positivas, la desigualdad de Cauchy–Schwarz, sistemas lineales, autovalores generalizados,
rango y condicionamiento de matrices, y regularización.

## Estado de la entrega

| Lo que pide la entrega | Dónde está | Estado |
|---|---|---|
| Problema, objetivo y alcance | `docs/desarrollo.md` (sec. 1) | Listo |
| Desarrollo matemático | `docs/desarrollo.md` (sec. 2 a 6 y 10) y `docs/desarrollo_matematico.pdf` | Listo |
| Implementación | `src/fisher.py` y `experiments/run_all.py` | Listo |
| Resultados y comprobaciones | `tests/test_fisher.py`, `results/`, `figures/` y `docs/desarrollo.md` (sec. 7) | Listo |
| Fuentes citadas (IEEE) | `docs/desarrollo.md` (Referencias) | Listo |
| Guion del video | `docs/guion_video.md` y `docs/guion_video.pdf` | Listo |
| Contribuciones de cada integrante | `docs/desarrollo.md` (sec. 11) | Listo |
| Caso aplicado con datos reales | `aplicacion/` y `docs/desarrollo.md` (sec. 8) | Listo |
| Limitaciones y cronograma | `docs/desarrollo.md` (sec. 9 y 12) | Listo |
| Declaración de uso de IA | `docs/declaracion_ia.md` | Listo |

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
Además, el 69.5 % de Fisher queda a 2 puntos del máximo teórico posible con estos datos (71.7 %, la regla de Bayes).

**Pocos datos** (30 datos y 40 variables, $S_W$ singular):

| Método | Exactitud en prueba |
|---|---|
| Sin regularizar ($\gamma = 0$) | 0.513 ± 0.035 |
| Pseudoinversa | 0.562 ± 0.027 |
| Regularizado, $\gamma$ elegido por validación cruzada | 0.619 ± 0.020 |
| Regularizado, mejor $\gamma$ de la rejilla ($\gamma = 10$) | 0.630 ± 0.010 |
| Diferencia de medias | 0.628 ± 0.010 |

Con $\gamma$ grande, $w$ tiende a la diferencia de medias, como predice la teoría.

**Datos reales: riesgo de acciones del S&P 500** (300 acciones de entrenamiento, 40 nuevas de prueba; cada acción
es $x = (\sigma, \beta, \mathrm{MDD}, \log(1+D/E))$ y es riesgosa si al año siguiente cayó más de 30 %):

| Dirección | Validación interna (20 particiones) | Validación externa (40 acciones) |
|---|---|---|
| Fisher | 0.696 ± 0.026 | 25 / 40 = 0.625 |
| PCA (primera componente) | 0.698 ± 0.036 | 25 / 40 = 0.625 |
| Diferencia de medias | 0.696 ± 0.036 | 25 / 40 = 0.625 |
| Siempre la clase mayoritaria | 0.578 | 0.575 |

Aquí Fisher no gana. La razón, con álgebra lineal: al escribir $w = \sum_i (u_i^\top d / \lambda_i)\, u_i$ con los
autovectores de $S_W$, el 93 % de $w$ viene de un solo autovector y las otras tres direcciones tienen más ruido que
señal; Fisher las divide entre autovalores pequeños y amplifica ese ruido. Regularizar lo lleva a la diferencia de
medias sin perder exactitud, y con 40 acciones de prueba (intervalo de confianza de 47 % a 76 %) ningún método se
distingue de otro. Está explicado en la sección 8 de `docs/desarrollo.md` y en [`aplicacion/README.md`](aplicacion/README.md).

## Organización de archivos

```
src/fisher.py            implementación: medias, S_W, dirección w, criterio J, clasificador, multiclase y regularización
tests/test_fisher.py     14 pruebas: ejemplo a mano, invarianzas, máximo de J, comparación con sklearn, rango de S_B,
                         S_W singular, medias iguales, gamma grande, intercambio de etiquetas y exactitud teórica
tests/test_aplicacion.py 6 pruebas del caso aplicado: misma dirección que src/fisher.py, orientación, estandarización,
                         separación estratificada, métricas y AUC, máximo de J
experiments/run_all.py   genera los experimentos sintéticos, tablas y figuras
results/  figures/       tablas (csv, json) y gráficos de los experimentos sintéticos
aplicacion/              caso aplicado con datos reales: scripts, datos guardados (datos/), tablas (resultados/) y
                         figuras (figuras/); su README explica cada archivo
verificar.py             corre pruebas, experimentos y caso aplicado, y comprueba que los resultados coinciden con los guardados
docs/                    desarrollo matemático y guion del video: el .pdf es para leer y el .md es su archivo fuente
                         declaracion_ia.md: declaración de uso de inteligencia artificial
```

## Cómo ejecutarlo

Se necesita Python 3 (lo probamos con Python 3.13).

```bash
pip install -r requirements.txt
python -m pytest -q tests          # corre las 20 pruebas
python experiments/run_all.py      # vuelve a generar results/ y figures/
python aplicacion/correr_todo.py   # vuelve a generar aplicacion/resultados/ y aplicacion/figuras/ (sin internet)
```

Para comprobarlo todo de una vez: `python verificar.py` corre lo anterior y compara cada tabla generada con la
guardada en el repositorio. Las semillas están fijas, así que se obtienen exactamente los mismos valores.
Volver a descargar los datos de Yahoo Finance es opcional (`pip install yfinance` y `python aplicacion/datos_300.py`).

## Datos

Los experimentos de `experiments/run_all.py` usan datos sintéticos generados en el mismo script:

- **Experimento A:** dos clases gaussianas con medias $(0,0)$ y $(1,0)$ y covarianza
  $\Sigma$ con varianzas 4 y 1 y covarianza 1.8 (en Python: `[[4, 1.8], [1.8, 1]]`). Semillas $2026 + k$, con $k = 0, \dots, 9$.
- **Experimento B:** 40 variables (2 informativas, 1 copia con ruido de la primera y 37 de ruido),
  15 datos de entrenamiento por clase. Semillas $2126 + k$, con $k = 0, \dots, 9$.

El caso aplicado usa datos reales de Yahoo Finance (precios diarios del 1-nov-2022 al 1-nov-2026 y balances),
guardados en `aplicacion/datos/` para que el análisis sea reproducible sin conexión.

## Detalles del método

- $w$ se obtiene resolviendo $(S_W + \lambda I) w = \mu_1 - \mu_2$ con `np.linalg.solve`, sin calcular la inversa, con $\lambda = \gamma \cdot \mathrm{tr}(S_W)/p$.
- El umbral es el punto medio de las medias proyectadas del entrenamiento (probabilidades previas iguales y costos simétricos).
- Las tres direcciones se comparan con la misma regla de umbral, y PCA se calcula solo con el entrenamiento.
- Entrenamiento y prueba se separan antes de ajustar, y $\gamma$ se elige con validación cruzada dentro del entrenamiento.

## Limitaciones

Los datos sintéticos tienen la misma covarianza en las dos clases, que es el caso favorable para Fisher, y
en el caso real Fisher no gana. No concluimos que Fisher sea siempre el mejor método: conviene cuando hay
una dirección de poca varianza con mucha señal, y en otro caso la diferencia de medias regularizada hace lo
mismo con menos riesgo. Las limitaciones del caso aplicado (sesgo de supervivencia, fecha del balance, prueba
pequeña) están en la sección 8.8 del documento.

## Referencias

Las referencias completas, en formato IEEE, están al final de
[`docs/desarrollo_matematico.pdf`](docs/desarrollo_matematico.pdf).
