---
title: "Discriminante lineal de Fisher: desarrollo matemático y resultados preliminares"
subtitle: "Álgebra Lineal Aplicada — Entrega parcial del proyecto"
author: "Elison Sanchez, Rodolfo Brandariz, Marcelo Bravo y Marcelo Torres"
date: "Noviembre 2026"
geometry: margin=2.3cm
fontsize: 12pt
mainfont: Arial
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{H}
---

# 1. Problema

Tenemos datos $x_i\in\mathbb R^p$ y sabemos a cuál de dos clases pertenece cada uno. La idea es reemplazar cada dato por un solo número, $y=w^\top x$, eligiendo $w$ de manera que en esa recta las dos clases queden separadas y, a la vez, cada clase quede lo más agrupada posible. Como veremos, lo único que importa es la dirección de $w$, no su tamaño.

**Objetivo.** Nuestro objetivo es programar el discriminante de Fisher para dos clases y comprobar cuatro cosas concretas. Primero, que el código da los mismos valores que el ejemplo hecho a mano ($w\propto(4,1)$, umbral 15 y $J=7$). Segundo, que la dirección que obtenemos es la misma que la de `LinearDiscriminantAnalysis` de scikit-learn. Tercero, que con datos gaussianos correlacionados Fisher acierta más en el conjunto de prueba que la diferencia de medias y que PCA, en promedio sobre 10 repeticiones. Y cuarto, que cuando $S_W$ es singular (más variables que datos) regularizar con $S_W+\lambda I$ mejora el resultado, siempre que $\lambda$ se elija mirando solo el entrenamiento. Todo esto se puede revisar con las pruebas de la carpeta `tests/` y con las tablas que genera `experiments/run_all.py`.

**Alcance.** Hicimos completo el caso de dos clases: la teoría, el código y los experimentos. El caso de más clases solo lo vemos en la teoría y en una prueba sobre el rango de $S_B$. Todos los datos son sintéticos, gaussianos y con la misma covarianza en las dos clases; no trabajamos con datos reales, con covarianzas distintas (eso llevaría a QDA) ni con clases que no se puedan separar con una recta.

**Conexión con el curso.** Todo el problema se resuelve con herramientas de álgebra lineal:

- proyectar un dato sobre una recta es un **producto interno**, $y=w^\top x$;
- el criterio de Fisher es un cociente de **formas cuadráticas** con matrices **simétricas semidefinidas positivas** ($S_W$ y $S_B$);
- el máximo se demuestra con la **desigualdad de Cauchy–Schwarz** (sección 3);
- calcular $w$ es resolver un **sistema lineal**, $S_Ww=\mu_1-\mu_2$, sin invertir la matriz;
- el caso de varias clases es un **problema de autovalores generalizados**, $S_Bw=\lambda S_Ww$ (sección 4);
- con pocos datos, $S_W$ tiene **rango** menor que $p$, es **singular** y está mal **condicionada**; se corrige con la **regularización** $S_W+\lambda I$ (secciones 6 y 7).

Notación para dos clases $\mathcal C_1,\mathcal C_2$: medias $\mu_k$, dispersiones $S_k=\sum_{x\in\mathcal C_k}(x-\mu_k)(x-\mu_k)^\top$, y dispersión intraclase $S_W=S_1+S_2$. Sea $d=\mu_1-\mu_2$.

**Por qué no basta la distancia entre medias.** Lo primero que se nos ocurre es buscar el $w$ que aleje más las medias proyectadas, es decir, maximizar $(w^\top d)^2$. El problema es que eso no tiene máximo: si duplicamos $w$, el valor se multiplica por cuatro. Podríamos fijar $\|w\|=1$, y entonces la respuesta sería $w\propto d$, pero esa dirección no toma en cuenta la forma de las nubes de puntos. Cuando las variables están correlacionadas, proyectar sobre $d$ puede hacer que las dos clases se encimen.

# 2. Criterio de Fisher

Como $w^\top S_Ww=\sum_k\sum_{x\in\mathcal C_k}(w^\top x-w^\top\mu_k)^2$, el denominador es la dispersión intraclase proyectada. Siguiendo a Fisher [1] (ver también [3], [5]), definimos
$$J(w)=\frac{(w^\top(\mu_1-\mu_2))^2}{w^\top S_Ww},\qquad w\neq 0 .$$

**Propiedades.** Hay dos cosas que conviene notar. La primera es que si multiplicamos $w$ por cualquier $c\ne0$, el numerador y el denominador se multiplican por $c^2$, así que $J(cw)=J(w)$: solo importa la dirección, y por eso $w$ y $-w$ dan lo mismo. La segunda es que $S_W$ es simétrica y semidefinida positiva, porque $w^\top S_Ww$ es una suma de cuadrados y nunca es negativa. Por ahora vamos a suponer que $S_W$ es definida positiva (invertible); en la sección 6 vemos qué pasa cuando no lo es.

# 3. Teorema y demostración

**Teorema.** Si $S_W\succ0$ y $d\neq0$, los maximizadores de $J$ son exactamente $w=c\,S_W^{-1}d$ con $c\neq0$, y
$$\max J = d^\top S_W^{-1}d .$$
Este valor es proporcional a la distancia de Mahalanobis al cuadrado entre las medias: si $\hat\Sigma=S_W/(N-2)$ es la covarianza agrupada, entonces $d^\top S_W^{-1}d=d^\top\hat\Sigma^{-1}d/(N-2)$.

*Demostración.* Como $S_W$ es simétrica, por el teorema espectral se puede escribir $S_W=Q\Lambda Q^\top$, con $Q$ ortogonal y $\Lambda$ diagonal con los autovalores. Como además es definida positiva, todos los autovalores son positivos, así que existe $S_W^{1/2}=Q\Lambda^{1/2}Q^\top$, que también es simétrica, definida positiva e invertible, y cumple $S_W^{1/2}S_W^{1/2}=S_W$. Sea $v=S_W^{1/2}w$. Entonces $w^\top d=v^\top S_W^{-1/2}d$ y $w^\top S_Ww=\|v\|^2$, de modo que
$$J=\frac{(v^\top a)^2}{\|v\|^2},\quad a=S_W^{-1/2}d .$$
Por Cauchy–Schwarz, $(v^\top a)^2\le\|v\|^2\|a\|^2$, con igualdad si y solo si $v=ca$. Luego $J\le\|a\|^2=d^\top S_W^{-1}d$ y la igualdad ocurre para $v=cS_W^{-1/2}d$, es decir $w=S_W^{-1/2}v=c\,S_W^{-1}d$. $\blacksquare$

**Casos.** Si $S_W=\sigma^2I$, entonces $w\propto d$ (diferencia de medias): no hay correlación que corregir. Si $d=0$, $J\equiv0$ y no hay dirección preferida; en ese caso el código lanza un error en lugar de devolver un vector sin sentido. En la práctica resolvemos el sistema $S_Ww=d$ (con `np.linalg.solve`, que usa una factorización LU) en lugar de calcular $S_W^{-1}$ y multiplicar: calcular la inversa hace más operaciones y acumula más error de redondeo, sobre todo cuando $S_W$ está mal condicionada, y no la necesitamos para nada más.

# 4. Formulación de Rayleigh y varias clases

Como $(w^\top d)^2=w^\top S_Bw$ con $S_B=dd^\top$ (rango $\le1$),
$$J(w)=\frac{w^\top S_Bw}{w^\top S_Ww},\qquad S_Bw=\lambda S_Ww .$$
Esto es un cociente de Rayleigh *generalizado*. Si $S_W$ es invertible, podemos hacer el mismo cambio de la demostración, $v=S_W^{1/2}w$, y el problema pasa a ser encontrar los autovalores de la matriz simétrica $S_W^{-1/2}S_BS_W^{-1/2}$. Esta matriz tiene los mismos autovalores que $S_W^{-1}S_B$, porque son semejantes. El valor máximo de $J$ es entonces el autovalor más grande. En el caso de dos clases se puede calcular directamente: como $S_B=dd^\top$ tiene rango 1, la matriz $S_W^{-1}S_B=S_W^{-1}dd^\top$ también tiene rango 1, así que tiene un solo autovalor distinto de cero. Para encontrarlo basta probar con $w=S_W^{-1}d$:
$$S_W^{-1}dd^\top(S_W^{-1}d)=(d^\top S_W^{-1}d)\,S_W^{-1}d ,$$
de modo que $S_W^{-1}d$ es autovector con autovalor $d^\top S_W^{-1}d$. Es justo el máximo y la dirección que obtuvimos en el teorema.

Para $K$ clases, con $n_k$, $\mu_k$, media global $\mu$ y $S_B=\sum_k n_k(\mu_k-\mu)(\mu_k-\mu)^\top$, las direcciones son los autovectores generalizados de mayores autovalores [2], [5]. Como $\sum_k n_k(\mu_k-\mu)=0$, $\operatorname{rango}(S_B)\le K-1$; hay a lo sumo $\min\{p,K-1\}$ direcciones útiles (verificado en `test_rank_SB_multiclass`).

# 5. Ejemplo calculable a mano

$\mathcal C_1=\{(1,1),(2,1),(2,2)\}$, $\mathcal C_2=\{(4,3),(5,3),(5,4)\}$.

| Paso | Resultado |
|---|---|
| Medias | $\mu_1=(5/3,4/3)$, $\mu_2=(14/3,10/3)$, $d=(-3,-2)$ |
| Dispersión | $S_1=S_2=\begin{pmatrix}2/3&1/3\\1/3&2/3\end{pmatrix}$, $S_W=\begin{pmatrix}4/3&2/3\\2/3&4/3\end{pmatrix}$ |
| Sistema $S_Ww=d$ | $\begin{pmatrix}4&2\\2&4\end{pmatrix}w=\begin{pmatrix}-9\\-6\end{pmatrix}\Rightarrow w=(-2,-\tfrac12)\propto(4,1)$ |
| Proyección $y=4x_1+x_2$ | $\mathcal C_1\mapsto\{5,9,10\}$, $\mathcal C_2\mapsto\{19,23,24\}$ |
| Medias proyectadas y umbral | $m_1=8$, $m_2=22$, $t=15$ |

Regla: $4x_1+x_2<15\Rightarrow\mathcal C_1$. Además $J_{\max}=d^\top S_W^{-1}d=7$, igual al valor calculado numéricamente. Es una verificación de la teoría, no una medida de generalización, porque se clasifican los mismos puntos con que se ajustó.

**Por qué el umbral va en el punto medio.** Fisher solo da la dirección $w$; el umbral es una decisión aparte. Usamos el punto medio $t=\tfrac12(w^\top\mu_1+w^\top\mu_2)$ porque es la regla óptima cuando las dos clases son gaussianas con la misma covarianza, tienen la misma probabilidad previa y equivocarse en un sentido cuesta lo mismo que en el otro. En ese caso (sección 9), la regla de Bayes compara $\delta_1(x)-\delta_2(x)=w^\top x-\tfrac12 w^\top(\mu_1+\mu_2)$ con cero, usando $w=\Sigma^{-1}(\mu_1-\mu_2)$; eso es exactamente comparar $w^\top x$ con el punto medio. Si las clases tuvieran probabilidades previas distintas, el umbral se movería en $\log(\pi_2/\pi_1)$.

![Ejemplo a mano](../figures/ej_mano.png){width=55%}

# 6. Singularidad y regularización

$S_W$ tiene rango a lo sumo $\min(p,N-K)$; es singular si $N-K<p$ o hay variables redundantes. Si $d=d_R+d_0$ con $d_R\in\operatorname{im}S_W$, $d_0\in\ker S_W$ y $d_0\ne0$, el cociente no tiene máximo finito (direcciones con dispersión intraclase cero pero separación positiva). Se regulariza:
$$S_{W,\lambda}=S_W+\lambda I,\quad w_\lambda=(S_W+\lambda I)^{-1}d,\quad \lambda=\gamma\,\operatorname{tr}(S_W)/p .$$
Escalamos $\lambda$ con $\operatorname{tr}(S_W)/p$, que es el promedio de los autovalores de $S_W$, es decir, la dispersión típica por variable. Así $\gamma$ no tiene unidades: si multiplicamos todos los datos por una constante $c$, $S_W$ y $\operatorname{tr}(S_W)$ se multiplican por $c^2$, $\lambda$ cambia igual y la dirección $w_\lambda$ no cambia. Además, $\gamma$ dice directamente cuánto pesa $\lambda I$ frente a $S_W$: con $\gamma=1$, el término agregado es del mismo tamaño que la dispersión promedio. Hay que tener claro que así ya no se maximiza el criterio original, sino uno modificado [5]. Cuando $\lambda$ es muy grande, el término $\lambda I$ domina sobre $S_W$ y $w_\lambda$ apunta casi en la dirección de $d$, como en el caso $S_W=\sigma^2 I$. El valor de $\gamma$ lo elegimos solo con validación dentro del entrenamiento.

# 7. Resultados preliminares

Todo es reproducible con `python experiments/run_all.py` (semillas $2026+k$, $k=0,\dots,9$).

**Experimento A: gaussianas correlacionadas.** $\mu_1=(0,0)$, $\mu_2=(1,0)$, $\Sigma=\begin{pmatrix}4&1.8\\1.8&1\end{pmatrix}$ (definida positiva: $\det=0.76$). 150 puntos por clase; división 50 % entrenamiento / 50 % prueba hecha antes del ajuste. Misma regla de umbral (punto medio de medias proyectadas de entrenamiento) para las tres direcciones. Para reportar $J$ usamos $\hat\Sigma=S_W/(N-2)$ en lugar de $S_W$. La razón es que $S_W$ es una suma sobre todos los puntos y crece con $N$, mientras que $\hat\Sigma$ es la covarianza agrupada y no depende del tamaño de la muestra; así el valor se puede comparar con el teórico. Con $\Delta=\mu_1-\mu_2=(-1,0)$, el máximo teórico es $\Delta^\top\Sigma^{-1}\Delta=1/0.76\approx1.32$.

| Método | $J$ (train) | Exactitud (prueba) | Ángulo con Fisher |
|---|---|---|---|
| Diferencia de medias | 0.27 ± 0.11 | 0.579 ± 0.034 | 60.1° ± 6.8° |
| **Fisher** | **1.19 ± 0.21** | **0.695 ± 0.032** | 0° |
| PCA (PC1) | 0.20 ± 0.12 | 0.559 ± 0.038 | 83.9° ± 2.8° |

(media ± desv. estándar en 10 repeticiones). Fisher gana en las 10 repeticiones de este escenario. La PC1 apunta a la dirección de máxima varianza (la diagonal de la nube), irrelevante para separar. La dirección de Fisher coincide con `LinearDiscriminantAnalysis` de scikit-learn [4], con `solver="lsqr"` (diferencia $\sim10^{-16}$). Matriz de confusión (repetición 0, fila = real): Fisher $\begin{pmatrix}52&20\\20&58\end{pmatrix}$, diferencia de medias $\begin{pmatrix}44&28\\32&46\end{pmatrix}$, PC1 $\begin{pmatrix}44&28\\35&43\end{pmatrix}$.

![Direcciones y frontera (repetición 0)](../figures/gauss_direcciones.png){width=60%}

![Histogramas de proyecciones en prueba](../figures/gauss_histogramas.png){width=95%}

**Experimento B: pocas muestras y variable redundante.** $p=40$ variables (2 informativas, 1 copia ruidosa de $x_1$, 37 de ruido), solo $N=30$ de entrenamiento y 1000 de prueba; $\mu_2=(1.5,0)$ en las dos primeras. Aquí $\operatorname{rango}(S_W)\le N-2=28<40$: en la repetición 0, $S_W$ tiene 12 autovalores nulos y número de condición $\sim10^{18}$.

| $\gamma$ | Exactitud media (prueba) | Desviación estándar |
|---|---|---|
| 0 | 0.513 | 0.035 |
| $10^{-4}$ | 0.547 | 0.041 |
| $10^{-3}$ | 0.548 | 0.041 |
| $10^{-2}$ | 0.552 | 0.039 |
| 0.1 | 0.571 | 0.035 |
| 0.3 | 0.589 | 0.025 |
| 1 | 0.605 | 0.020 |
| 3 | 0.626 | 0.015 |
| 10 | 0.630 | 0.010 |
| 30 | 0.629 | 0.011 |
| 100 | 0.629 | 0.010 |

Como referencia, en las mismas 10 repeticiones medimos también:

| Método de comparación | Exactitud media (prueba) | Desviación estándar |
|---|---|---|
| Pseudoinversa, $w=S_W^{+}d$ (sin regularizar) | 0.562 | 0.027 |
| Diferencia de medias, $w\propto d$ | 0.628 | 0.010 |
| $\gamma$ elegido por validación cruzada | 0.619 | 0.020 |

Sin regularizar ($\gamma=0$), el sistema es numéricamente inestable y la exactitud media es 0.513, prácticamente al azar. Por eso agregamos al código un aviso que salta cuando el número de condición de $S_W$ pasa de $10^{12}$. Con la pseudoinversa la exactitud sube a 0.562. Sin embargo, el material del curso [5] advierte que la pseudoinversa no resuelve el criterio original cuando $S_W$ es singular, porque solo escoge una de muchas direcciones posibles; por eso la incluimos únicamente como comparación.

A medida que aumenta $\gamma$, la exactitud sube hasta 0.630 y después se queda casi igual. Esto era de esperar por lo que vimos en la sección 6: con $\lambda$ grande, $w_\lambda$ se acerca a la dirección de $d$, y efectivamente los valores grandes de $\gamma$ dan casi lo mismo que la diferencia de medias (0.628). Nuestra interpretación es que con solo 30 datos $S_W$ queda muy mal estimada, y conviene apoyarse casi solo en las medias. Ojo: estos valores son de prueba y los mostramos solo para describir el comportamiento; no los usamos para elegir $\gamma$.

Cuando elegimos $\gamma$ con validación cruzada de cinco pliegues, usando solo el entrenamiento, la exactitud media en prueba es 0.619 (desviación estándar 0.020). El $\gamma$ elegido cambió bastante de una repetición a otra (desde $10^{-4}$ hasta 100), lo cual tiene sentido: con tan pocos datos, cada pliegue de validación tiene apenas unos 6 puntos y la estimación es ruidosa.

![Regularización, pseudoinversa y diferencia de medias](../figures/regularizacion.png){width=60%}

# 8. Limitaciones y trabajo restante

- Los datos son sintéticos y cumplen el supuesto de covarianza común; falta probar covarianzas distintas (donde aparece QDA), clases no separables linealmente y etiquetas ruidosas. **No se concluye que Fisher domine siempre.**
- En el experimento B el mejor desempeño se parece al de la diferencia de medias, así que con $N<p$ la ventaja de Fisher depende de cuánta regularización se use.
- Pendiente: un conjunto de datos real y el caso multiclase con datos reales.

# 9. Relación con LDA probabilístico

Si $X\mid Y=k\sim\mathcal N(\mu_k,\Sigma)$ con priors $\pi_k$, la regla de Bayes maximiza $\delta_k(x)=x^\top\Sigma^{-1}\mu_k-\tfrac12\mu_k^\top\Sigma^{-1}\mu_k+\log\pi_k$, Esta función es afín en $x$, así que las fronteras entre clases son rectas (o hiperplanos). Para dos clases, el vector normal a la frontera es $\Sigma^{-1}(\mu_1-\mu_2)$. Si estimamos $\Sigma$ con la covarianza agrupada, que es proporcional a $S_W$, obtenemos la misma dirección que con el criterio de Fisher. Es decir, partiendo de una idea geométrica (Fisher) y de una probabilística (Bayes) se llega al mismo resultado [2], [3].

# 10. Organización del equipo y contribuciones

El proyecto se desarrolla de manera colaborativa. Las responsabilidades iniciales se distribuyen de la siguiente manera:

| Integrante | Responsabilidad principal | Actividades |
|---|---|---|
| Elison Sanchez | Fundamentos matemáticos | Formulación del criterio de Fisher, demostración matemática y revisión de la notación. |
| Rodolfo Brandariz | Desarrollo matemático aplicado | Resolución del ejemplo manual, cálculo de matrices de dispersión y explicación de las proyecciones. |
| Marcelo Bravo | Implementación computacional | Desarrollo y revisión del código Python, implementación del algoritmo y comparación con PCA. |
| Marcelo Torres | Experimentación y documentación | Experimentos de regularización, análisis de resultados, gráficos y documentación del repositorio. |

# 11. Cronograma de actividades

| Actividad | Responsable(s) | Fecha límite | Estado |
|---|---|---|---|
| Revisión y corrección matemática | Elison Sanchez y Rodolfo Brandariz | 2/11/2026 | En revisión |
| Verificación de la implementación y pruebas | Marcelo Bravo | 2/11/2026 | En revisión |
| Actualización de experimentos y resultados | Marcelo Bravo y Marcelo Torres | 2/11/2026 | En revisión |
| Interpretación y discusión de resultados | Todos los integrantes | 2/11/2026 | En revisión |
| Actualización de documentos y referencias | Rodolfo Brandariz y Marcelo Torres | 7/11/2026 | En revisión |
| Preparación y grabación del video | Todos los integrantes | 7/11/2026 | Pendiente |
| Verificación final del repositorio y entrega | Todos los integrantes | 9/11/2026 | Pendiente |

# Referencias

[1] R. A. Fisher, "The use of multiple measurements in taxonomic problems," *Annals of Eugenics*, vol. 7, no. 2, pp. 179–188, 1936, doi: 10.1111/j.1469-1809.1936.tb02137.x.

[2] T. Hastie, R. Tibshirani y J. Friedman, *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*, 2nd ed. New York, NY, USA: Springer, 2009, ch. 4.

[3] C. M. Bishop, *Pattern Recognition and Machine Learning*. New York, NY, USA: Springer, 2006, sec. 4.1.4.

[4] scikit-learn developers, "Linear and Quadratic Discriminant Analysis," *scikit-learn documentation*. [En línea]. Disponible: https://scikit-learn.org/stable/modules/lda_qda.html. [Accedido: 2 nov. 2026].

[5] O. Velásquez Castañón, "Separación de clases con el discriminante de Fisher: proyecciones, cocientes de Rayleigh y clasificación," material de referencia, Álgebra Lineal Aplicada 2026-2, Universidad del Pacífico, Lima, Perú, oct. 2026.
