# Declaración de uso de inteligencia artificial

Según la sección 9 de las bases del proyecto, declaramos el uso de herramientas de inteligencia artificial como apoyo. El grupo es responsable de la exactitud de todo el contenido entregado, y cada integrante puede explicar y defender las partes elaboradas con este apoyo.

**Herramienta:** Claude (Anthropic), modelo Claude Opus 5.5.

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

## Procedimientos de verificación

- Ejecutamos las 14 pruebas de `tests/` con las versiones de `requirements.txt` y todas pasan.
- Ejecutamos `experiments/run_all.py` desde una copia limpia del repositorio y obtuvimos exactamente los mismos resultados y figuras guardados en `results/` y `figures/`.
- Recalculamos a mano el ejemplo de la sección 5 de `docs/desarrollo.md` ($w\propto(4,1)$, umbral 15, $J=7$) y lo comparamos con el código.
- Comparamos la dirección de Fisher con `LinearDiscriminantAnalysis` de scikit-learn (diferencia del orden de $10^{-16}$).
- Comparamos las exactitudes obtenidas con los valores teóricos de la distribución que generó los datos.
