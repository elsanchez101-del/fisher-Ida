---
title: "Guion del video: Discriminante lineal de Fisher"
geometry: margin=2cm
fontsize: 12pt
mainfont: Arial
header-includes:
  - \usepackage{titling}
  - \setlength{\droptitle}{-2.2cm}
  - \posttitle{\par\end{center}\vspace{-1.2cm}}
  - \usepackage{titlesec}
  - \titlespacing*{\section}{0pt}{10pt}{4pt}
  - \titlespacing*{\subsection}{0pt}{8pt}{2pt}
---

# Parte 1: Elison Sanchez (0:00 a 3:00)

## Título (0:00 a 0:25)

**Quién:** Elison Sanchez  
**Pantalla:** Diapositiva con el tema, el título y los integrantes.

Hola, qué tal. Somos Elison Sanchez, Rodolfo Brandariz, Marcelo Bravo y Marcelo Torres, y en este video
les vamos a contar nuestro proyecto sobre el discriminante lineal de Fisher, que sirve para separar dos clases.

## El problema (0:25 a 1:20)

**Quién:** Elison Sanchez  
**Pantalla:** Nube de dos clases correlacionadas y su proyección sobre la recta de las medias.

Imaginen que tenemos datos de dos grupos y cada dato tiene varias variables. Lo que queremos es
quedarnos con un solo número por dato, $y = w^\top x$, pero sin que los grupos se mezclen. Lo primero que
se nos ocurrió fue proyectar sobre la recta que une las medias, $d = \mu_1 - \mu_2$. Pero como ven acá,
cuando las variables están correlacionadas, las dos clases terminan encimándose. Y si solo maximizamos $(w^\top d)^2$, basta con agrandar $w$ para que crezca sin límite.

## El criterio de Fisher (1:20 a 2:05)

**Quién:** Elison Sanchez  
**Pantalla:** Diapositiva con $J(w)$.

Entonces, lo que propone Fisher es dividir la separación de las medias entre qué tan dispersa queda cada clase:
$$J(w) = \frac{(w^\top d)^2}{w^\top S_W w}.$$
Abajo tenemos $S_W$, la dispersión dentro de las clases. Ese denominador es una suma de cuadrados, así que
nunca sale negativo. Y algo importante: si multiplicamos $w$ por cualquier número, $J$ no cambia. O sea,
lo único que nos importa es hacia dónde apunta $w$.

## Demostración (2:05 a 3:00)

**Quién:** Elison Sanchez  
**Pantalla:** Pasos de la demostración.

Para encontrar el máximo hacemos el cambio $v = S_W^{1/2} w$. Con eso el cociente queda
$(v^\top a)^2 / \lVert v \rVert^2$, donde $a = S_W^{-1/2} d$. Por Cauchy–Schwarz, esto es máximo cuando
$v$ es paralelo a $a$, y si regresamos a $w$ nos queda:
$$w \propto S_W^{-1}(\mu_1 - \mu_2).$$
Básicamente, Fisher toma la dirección de las medias y la corrige con $S_W^{-1}$. Y si los datos no
estuvieran correlacionados, $S_W$ sería un múltiplo de la identidad y volveríamos a la recta de las medias.

# Parte 2: Rodolfo Brandariz (3:00 a 5:00)

## Cociente de Rayleigh y varias clases (3:00 a 3:40)

**Quién:** Rodolfo Brandariz  
**Pantalla:** Diapositiva con $S_B w = \lambda S_W w$.

Gracias, Elison. Ahora, este mismo criterio también se puede escribir de otra forma. Si llamamos
$S_B = dd^\top$, nos queda $w^\top S_B w / w^\top S_W w$, que es un cociente de Rayleigh generalizado, y sus
puntos críticos cumplen $S_B w = \lambda S_W w$. Esto nos sirve para el caso de más clases: con $K$ clases,
$S_B$ tiene rango a lo más $K-1$, entonces solo podemos sacar hasta $K-1$ direcciones útiles.

## Ejemplo a mano (3:40 a 5:00)

**Quién:** Rodolfo Brandariz  
**Pantalla:** Tabla con los cálculos y la figura `ej_mano.png`.

Para ver que todo esto funciona, hicimos un ejemplo chiquito a mano, con tres puntos por clase. Primero
sacamos las medias y la matriz de dispersión, que nos dio
$S_W = \begin{pmatrix} 4/3 & 2/3 \\ 2/3 & 4/3 \end{pmatrix}$. Después resolvimos $S_W w = d$ y nos salió $w$
proporcional a $(4, 1)$. Si proyectamos, la primera clase cae en 5, 9 y 10, y la segunda en 19, 23 y 24.
El umbral es el punto medio, 15, y el $J$ máximo es 7, que es justo lo que nos da el código. Eso sí, esto solo comprueba que los cálculos están bien, no qué tan bien clasifica datos nuevos.

# Parte 3: Marcelo Bravo (5:00 a 7:00)

## Implementación y pruebas (5:00 a 5:40)

**Quién:** Marcelo Bravo  
**Pantalla:** Archivo `src/fisher.py` y ejecución de `pytest`.

Bueno, ahora pasamos al código. Programamos todo en Python con NumPy. Para sacar $w$ resolvemos el sistema
con `solve`, sin calcular la inversa. Además armamos pruebas automáticas, 20 en total: que salga el ejemplo a mano, que $J$ no
cambie al escalar $w$, que ninguna de 500 direcciones al azar le gane a Fisher, y que nuestro resultado
coincida con el de scikit-learn. Y coincide: la diferencia es del orden de $10^{-16}$.

## Experimento con datos correlacionados (5:40 a 7:00)

**Quién:** Marcelo Bravo  
**Pantalla:** `gauss_direcciones.png`, histogramas y tabla de resultados.

Después lo probamos con datos más grandes. Generamos dos clases gaussianas con covarianza
$\Sigma = \begin{pmatrix} 4 & 1.8 \\ 1.8 & 1 \end{pmatrix}$, separamos entrenamiento y prueba antes de
ajustar, y lo repetimos 10 veces. Comparamos tres direcciones usando la misma regla de umbral. En prueba,
Fisher acertó en promedio 69.5 %, la diferencia de medias 57.9 % y PCA 55.9 %, y Fisher ganó en las 10
repeticiones. Lo de PCA es curioso: elige la dirección donde los datos varían más, que acá es la diagonal
de la nube, pero esa dirección no separa las clases porque PCA ni siquiera mira las etiquetas.

# Parte 4: Marcelo Torres (7:00 a 10:50)

## Regularización con pocos datos (7:00 a 7:45)

**Quién:** Marcelo Torres  
**Pantalla:** `regularizacion.png`.

Ahora, ¿qué pasa cuando hay pocos datos? Probamos con 30 datos y 40 variables, y ahí $S_W$ se vuelve
singular. Sin regularizar, el método acierta 51 %, como lanzar una moneda. Si usamos $S_W + \lambda I$, la
matriz ya se puede invertir y la exactitud sube hasta 63 %. Y con $\lambda$ grande el resultado queda casi
igual al de la diferencia de medias, que es lo que dice la teoría: cuando $\lambda$ crece, $w$ tiende a $d$.
Quédense con esta idea, porque va a volver a aparecer.

## Caso aplicado: acciones del S&P 500 (7:45 a 8:35)

**Quién:** Marcelo Torres  
**Pantalla:** Tabla de las cuatro variables y tabla de la validación externa.

Hasta acá todo era con datos inventados, así que quisimos probar con datos reales. Tomamos 300 acciones del
S&P 500 y para cada una calculamos cuatro cosas con los precios de tres años: volatilidad, beta, la caída
máxima que tuvo y su apalancamiento. La pregunta es si en el año siguiente la acción cayó más de 30 %.
Entrenamos con esas 300 y probamos con 40 acciones nuevas que el modelo nunca vio. ¿Y qué pasó? Fisher,
PCA y la diferencia de medias acertaron exactamente lo mismo: 25 de 40. Fisher no ganó. Y eso, en vez de
esconderlo, es lo que quisimos entender.

## Por qué Fisher no gana aquí (8:35 a 9:50)

**Quién:** Marcelo Torres  
**Pantalla:** `espectro_w.png` y luego `camino_regularizacion.png`.

La clave está en escribir $w$ con el teorema espectral. Si $S_W$ tiene autovectores $u_i$ y autovalores
$\lambda_i$, entonces
$$w = \sum_i \frac{u_i^\top d}{\lambda_i}\, u_i .$$
O sea, Fisher divide entre el autovalor: le da más peso a las direcciones donde las clases están menos
dispersas. En los datos sintéticos eso era justo lo que hacía falta. Pero acá las tres variables de mercado
están muy correlacionadas, y resulta que el 93 % de $w$ viene de un solo autovector, el de "riesgo general".
Las otras tres direcciones tienen más ruido que señal, y Fisher las divide entre autovalores pequeños, así
que amplifica ese ruido. Lo que Fisher agrega sobre la diferencia de medias es, básicamente, ruido.

Y acá vuelve la regularización. Si recorremos $\lambda$ desde cero hasta infinito, $w$ gira 18 grados hasta
coincidir con la diferencia de medias, y la exactitud no se mueve. Regularizar no cuesta nada. Además,
con 40 acciones de prueba el intervalo de confianza va de 47 % a 76 %, así que ningún método se puede
declarar mejor que otro.

## Tres clases y conclusiones (9:50 a 10:35)

**Quién:** Marcelo Torres  
**Pantalla:** `tres_clases_span.png` y diapositiva de conclusiones.

Por último, con tres niveles de riesgo, $S_B$ tiene rango 2 y el problema de autovalores generalizado nos da
un plano, el generado por $w_1$ y $w_2$, donde las tres clases quedan lo más separadas posible. Así podemos
ver en dos dimensiones lo que pasa en cuatro.

En resumen: Fisher llega casi al óptimo cuando hay una dirección de poca varianza con mucha señal. Cuando
no la hay, como en las acciones, empata con la diferencia de medias y conviene regularizar. Las limitaciones
están en el documento: usamos la lista actual del índice, que deja fuera a las empresas que salieron, un solo
año de etiqueta y una prueba pequeña.

## Cierre (10:35 a 10:50)

**Quién:** Marcelo Torres  
**Pantalla:** Referencias principales y enlace al repositorio.

Todo el código, los datos y las referencias están en nuestro repositorio de GitHub; se corre con un solo
comando. Muchas gracias por vernos.
