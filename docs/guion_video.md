---
title: "Guion del video — Discriminante lineal de Fisher (≈10 min)"
geometry: margin=2.3cm
fontsize: 11pt
---

Reparto sugerido para un grupo de 3 (ajustar): **A** = problema y teoría, **B** = ejemplo y código, **C** = experimentos y conclusiones. Ritmo: ~130 palabras/min.

| Min | Quién | Pantalla | Qué decir |
|------|---|---------|------------------------------|
| 0:00–1:00 | A | Diapositiva título; nube de dos clases | «Tenemos datos en $\mathbb R^p$ con dos clases. Queremos reducir a una sola coordenada $y=w^\top x$ sin perder la separación. ¿Cómo elegir $w$?» Presentar integrantes y qué entregamos: teoría, código, experimentos. |
| 1:00–2:00 | A | Figura de proyección sobre $d=\mu_1-\mu_2$ con correlación | «Primera idea: proyectar sobre la diferencia de medias. Falla si las nubes están correlacionadas: las proyecciones se solapan. Además $(w^\top d)^2$ crece al escalar $w$, no define el problema.» |
| 2:00–3:30 | A | $J(w)$ y $S_W$ en pizarra/diapositiva | «Fisher: maximizar separación entre medias sobre dispersión interna. $w^\top S_Ww$ es la suma de cuadrados de las proyecciones centradas. $J$ no cambia al escalar $w$, ni con $-w$.» |
| 3:30–5:00 | A | Demostración Cauchy–Schwarz | «Con $v=S_W^{1/2}w$, $J=(v^\top a)^2/\|v\|^2$. Cauchy–Schwarz da el máximo cuando $v\propto a$, es decir $w\propto S_W^{-1}d$. Máximo = $d^\top S_W^{-1}d$. Si $S_W=\sigma^2I$ se recupera la diferencia de medias.» Mencionar cociente de Rayleigh generalizado, $S_Bw=\lambda S_Ww$, y $K-1$ direcciones. |
| 5:00–6:15 | B | Ejemplo a mano (tabla) + `ej_mano.png` | «Seis puntos. $S_W=\begin{pmatrix}4/3&2/3\\2/3&4/3\end{pmatrix}$, resolvemos $S_Ww=d$, sale $w\propto(4,1)$. Proyecciones 5, 9, 10 contra 19, 23, 24; umbral 15; $J=7$ coincide con el código.» |
| 6:15–7:00 | B | `src/fisher.py` y `pytest` en terminal | «Implementado desde cero con `solve`, sin invertir. 8 tests: ejemplo a mano, invarianza, $J$ máximo contra direcciones aleatorias, rango de $S_B$, comparación con scikit-learn (diferencia $10^{-16}$).» |
| 7:00–8:30 | C | `gauss_direcciones.png`, tabla A, histogramas | «Gaussianas con $\Sigma=\begin{pmatrix}4&1.8\\1.8&1\end{pmatrix}$, 10 repeticiones, train/test antes de ajustar, misma regla de umbral. Exactitud: Fisher 0.695, diferencia de medias 0.579, PCA 0.559. PCA apunta a la mayor varianza, que no separa.» |
| 8:30–9:15 | C | `regularizacion.png` | «Con 30 muestras y 40 variables $S_W$ es singular (12 autovalores nulos). Sin regularizar: 0.51. Con $S_W+\lambda I$ sube hasta 0.63. Elegimos $\gamma$ solo con validación dentro del entrenamiento.» |
| 9:15–10:00 | C | Diapositiva de limitaciones | «Limitaciones: datos sintéticos con covarianza común, que favorecen a Fisher; no concluimos que domine siempre. Falta: datos reales, multiclase, covarianzas distintas (QDA), ampliar la rejilla de $\gamma$. Repositorio en GitHub: (enlace).» |

## Preguntas de defensa (respuestas breves)

1. **¿Por qué no basta la distancia entre medias proyectadas?** Se maximiza escalando $w$; normalizando, ignora la forma de las clases.
2. **¿Rol de $S_W$ y por qué es semidefinida positiva?** Mide la dispersión interna proyectada; $w^\top S_Ww$ es suma de cuadrados.
3. **¿Por qué $w$ y $-w$ equivalen?** $J(cw)=J(w)$. Solo se invierte el sentido del umbral.
4. **¿Y si las medias coinciden o $S_W$ es singular?** Con $d=0$, $J\equiv0$. Con $S_W$ singular se regulariza, y se maximiza un criterio modificado.
5. **¿Cómo se elige el umbral?** Punto medio de medias proyectadas: justificado con covarianza común, priors iguales y costos simétricos.
6. **¿Por qué no usar etiquetas de prueba para elegir $w$ o $\gamma$?** Sesgaría la estimación del error de generalización.
