# Declaración de uso de inteligencia artificial

Según la sección 9 de las bases del proyecto, declaramos el uso de herramientas de inteligencia artificial como apoyo. El grupo es responsable de la exactitud de todo el contenido entregado, y cada integrante puede explicar y defender las partes elaboradas con este apoyo.

**Herramienta:** Claude (Anthropic), modelos Claude Opus 5.5 (entrega parcial) y Claude Fable 5.1 (entrega final).

## 1. Apoyo en la investigación

- **Finalidad:** entender el discriminante de Fisher, elegir el tema del proyecto y revisar que el trabajo cumpliera las bases y la rúbrica del curso.
- **Instrucciones relevantes:** "¿Cuál tema se adapta mejor al proyecto?", "¿En qué consiste el de Fisher?", "Explícame de qué va el proyecto que hemos desarrollado", "Verifica todo el trabajo punto por punto", "Evalúa esto" (con cada criterio de la rúbrica).
- **Partes del trabajo:** comparación entre los temas propuestos; explicación del problema, del criterio de Fisher y de los resultados; revisión del repositorio frente a la rúbrica; propuesta de justificaciones para el desarrollo matemático (existencia de $S_W^{1/2}$, autovalor de una matriz de rango 1, umbral en el punto medio, escala de $\lambda$) y de la comparación con la exactitud teórica $\Phi(\delta/2\sigma)$.

## 2. Apoyo en la elaboración de código

- **Finalidad:** revisar y mejorar la implementación y los experimentos.
- **Instrucciones relevantes:** "Revisa el código, ¿no se puede hacer alguna mejora ahí?", "Aplica todo".
- **Partes del trabajo:**
  - `src/fisher.py`: error cuando las medias coinciden, aviso cuando $S_W$ es numéricamente singular, opción de pseudoinversa para comparar, revisión del denominador de $J$ y la función `theoretical_accuracy`.
  - `tests/test_fisher.py`: 6 pruebas nuevas (medias iguales, $\gamma$ grande, intercambio de etiquetas, $J$ indefinido, $S_W$ singular y exactitud teórica).
  - `experiments/run_all.py`: rejilla de $\gamma$ ampliada hasta 100, pseudoinversa y diferencia de medias como referencia en el experimento B, y cálculo de la exactitud teórica.
  - `requirements.txt`: versiones fijadas.

## 3. Apoyo en la redacción de la documentación

- **Finalidad:** ordenar y redactar con mayor claridad la documentación de la entrega parcial.
- **Partes del trabajo:** formato IEEE de las referencias; redacción del guion del video para los cuatro integrantes; README con el estado de la entrega.

## 4. Apoyo en el caso aplicado (entrega final)

- **Finalidad:** integrar al repositorio el caso de riesgo de acciones, hecho por un integrante en scripts separados, y analizar por qué Fisher no gana en datos reales.
- **Instrucciones relevantes:** "Evalúa este proyecto y las mejoras que quiero implementar", "Explícame mejor mi aporte", "¿Ahora qué falta?".
- **Partes del trabajo:**
  - `aplicacion/`: reorganización de los scripts originales (`datos_300.py`, `escenarios_lda_pca.py`, `validacion_externa.py`) en un módulo común `comun.py`, rutas relativas, datos guardados en `datos/` para que todo corra sin internet, y `correr_todo.py`. Se comprobó que los scripts reorganizados dan exactamente los mismos números que los originales.
  - Análisis nuevos propuestos y programados con apoyo de la herramienta: `analisis_espectral.py` (descomposición $w=\sum_i (u_i^\top d/\lambda_i)u_i$, señal/ruido por componente y camino de regularización), `comparacion_sin_umbral.py` (AUC, intervalos de Wilson, McNemar), `estabilidad_pesos.py` (bootstrap) y `tres_clases.py` (Fisher multiclase y plano $\operatorname{span}(w_1,w_2)$).
  - `tests/test_aplicacion.py`: 6 pruebas nuevas.
  - `verificar.py`: comprobación de reproducibilidad con un solo comando.
  - Redacción de la sección 8 de `docs/desarrollo.md`, del bloque correspondiente del guion y de la actualización del README, a partir de los resultados obtenidos por el grupo; corrección de inconsistencias del borrador original (notación, número de pruebas, factor $N-2$ en la distancia de Mahalanobis).
- Los datos fueron descargados y los scripts ejecutados por los integrantes en sus propias computadoras; la interpretación de los resultados fue discutida con la herramienta y cada integrante puede explicarla.

## Procedimientos de verificación

- Ejecutamos las 20 pruebas de `tests/` con las versiones de `requirements.txt` y todas pasan.
- Ejecutamos `experiments/run_all.py` desde una copia limpia del repositorio y obtuvimos exactamente los mismos resultados y figuras guardados en `results/` y `figures/`.
- Recalculamos a mano el ejemplo de la sección 5 de `docs/desarrollo.md` ($w\propto(4,1)$, umbral 15, $J=7$) y lo comparamos con el código.
- Comparamos la dirección de Fisher con `LinearDiscriminantAnalysis` de scikit-learn (diferencia del orden de $10^{-16}$).
- Comparamos las exactitudes obtenidas con los valores teóricos de la distribución que generó los datos.
- En el caso aplicado, dos ejecuciones seguidas de `aplicacion/correr_todo.py` dan tablas idénticas; la dirección de Fisher coincide con scikit-learn (coseno 1); la suma espectral reconstruye $\hat\Sigma^{-1}d$ (comprobado con `assert` en el script); el AUC coincide con `roc_auc_score` de scikit-learn; y `python verificar.py` repite todo lo anterior y compara con las tablas guardadas.
