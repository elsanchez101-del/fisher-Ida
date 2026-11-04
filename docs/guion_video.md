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

# Parte 1: Elison Sanchez (0:00 a 3:15)

## Título (0:00 a 0:30)

**Quién:** Elison Sanchez  
**Pantalla:** Diapositiva con el tema, el título y los integrantes.

Hola, qué tal. Somos Elison Sanchez, Rodolfo Brandariz, Marcelo Bravo y Marcelo Torres, y en este video
les vamos a contar nuestro proyecto sobre el discriminante lineal de Fisher, que sirve para separar dos clases.

## El problema (0:30 a 1:30)

**Quién:** Elison Sanchez  
**Pantalla:** Nube de dos clases correlacionadas y su proyección sobre la recta de las medias.

Imaginen que tenemos datos de dos grupos y cada dato tiene varias variables. Lo que queremos es
quedarnos con un solo número por dato, $y = w^\top x$, pero sin que los grupos se mezclen. Lo primero que
se nos ocurrió fue proyectar sobre la recta que une las medias, $d = \mu_1 - \mu_2$. Pero como ven acá,
cuando las variables están correlacionadas, las dos clases terminan encimándose. Y otro detalle: si solo
maximizamos $(w^\top d)^2$, basta con hacer $w$ más grande para que ese valor crezca todo lo que queramos.

## El criterio de Fisher (1:30 a 2:15)

**Quién:** Elison Sanchez  
**Pantalla:** Diapositiva con $J(w)$.

Entonces, lo que propone Fisher es dividir la separación de las medias entre qué tan dispersa queda cada clase:
$$J(w) = \frac{(w^\top d)^2}{w^\top S_W w}.$$
Abajo tenemos $S_W$, la dispersión dentro de las clases. Ese denominador es una suma de cuadrados, así que
nunca sale negativo. Y algo importante: si multiplicamos $w$ por cualquier número, $J$ no cambia. O sea,
lo único que nos importa es hacia dónde apunta $w$.

## Demostración (2:15 a 3:15)

**Quién:** Elison Sanchez  
**Pantalla:** Pasos de la demostración.

Para encontrar el máximo hacemos el cambio $v = S_W^{1/2} w$. Con eso el cociente queda
$(v^\top a)^2 / \lVert v \rVert^2$, donde $a = S_W^{-1/2} d$. Por Cauchy–Schwarz, esto es máximo cuando
$v$ es paralelo a $a$, y si regresamos a $w$ nos queda:
$$w \propto S_W^{-1}(\mu_1 - \mu_2).$$
Básicamente, Fisher toma la dirección de las medias y la corrige con $S_W^{-1}$. Y si los datos no
estuvieran correlacionados, $S_W$ sería un múltiplo de la identidad y volveríamos a la recta de las medias.

# Parte 2: Rodolfo Brandariz (3:15 a 5:30)

## Cociente de Rayleigh y varias clases (3:15 a 4:00)

**Quién:** Rodolfo Brandariz  
**Pantalla:** Diapositiva con $S_B w = \lambda S_W w$.

Gracias, Elison. Ahora, este mismo criterio también se puede escribir de otra forma. Si llamamos
$S_B = dd^\top$, nos queda $w^\top S_B w / w^\top S_W w$, que es un cociente de Rayleigh generalizado, y sus
puntos críticos cumplen $S_B w = \lambda S_W w$. Esto nos sirve para el caso de más clases: con $K$ clases,
$S_B$ tiene rango a lo más $K-1$, entonces solo podemos sacar hasta $K-1$ direcciones útiles.

## Ejemplo a mano (4:00 a 5:30)

**Quién:** Rodolfo Brandariz  
**Pantalla:** Tabla con los cálculos y la figura `ej_mano.png`.

Para ver que todo esto funciona, hicimos un ejemplo chiquito a mano, con tres puntos por clase. Primero
sacamos las medias y la matriz de dispersión, que nos dio
$S_W = \begin{pmatrix} 4/3 & 2/3 \\ 2/3 & 4/3 \end{pmatrix}$. Después resolvimos $S_W w = d$ y nos salió $w$
proporcional a $(4, 1)$. Si proyectamos, la primera clase cae en 5, 9 y 10, y la segunda en 19, 23 y 24.
El umbral es el punto medio, 15, y el $J$ máximo es 7, que es justo lo que nos da el código. Eso sí, esto
solo comprueba que los cálculos están bien; no nos dice qué tan bien clasifica datos nuevos, porque estamos
usando los mismos puntos con los que ajustamos.

# Parte 3: Marcelo Bravo (5:30 a 7:45)

## Implementación y pruebas (5:30 a 6:15)

**Quién:** Marcelo Bravo  
**Pantalla:** Archivo `src/fisher.py` y ejecución de `pytest`.

Bueno, ahora pasamos al código. Programamos todo en Python con NumPy. Para sacar $w$ resolvemos el sistema
con `solve`, sin calcular la inversa. Además armamos 14 pruebas: que salga el ejemplo a mano, que $J$ no
cambie al escalar $w$, que ninguna de 500 direcciones al azar le gane a Fisher, que el código avise si las
medias son iguales o si $S_W$ es singular, y que nuestro resultado coincida con el de scikit-learn. Y
coincide: la diferencia es del orden de $10^{-16}$.

## Experimento con datos correlacionados (6:15 a 7:45)

**Quién:** Marcelo Bravo  
**Pantalla:** `gauss_direcciones.png`, histogramas y tabla de resultados.

Después lo probamos con datos más grandes. Generamos dos clases gaussianas con covarianza
$\Sigma = \begin{pmatrix} 4 & 1.8 \\ 1.8 & 1 \end{pmatrix}$, separamos entrenamiento y prueba antes de
ajustar, y lo repetimos 10 veces. Comparamos tres direcciones usando la misma regla de umbral. En prueba,
Fisher acertó en promedio 69.5 %, la diferencia de medias 57.9 % y PCA 55.9 %, y Fisher ganó en las 10
repeticiones. Lo de PCA es curioso: elige la dirección donde los datos varían más, que acá es la diagonal
de la nube, pero esa dirección no separa las clases porque PCA ni siquiera mira las etiquetas.

# Parte 4: Marcelo Torres (7:45 a 10:00)

## Regularización (7:45 a 8:45)

**Quién:** Marcelo Torres  
**Pantalla:** `regularizacion.png`.

Por último, nos preguntamos qué pasa cuando hay pocos datos. Probamos con 30 datos y 40 variables, y ahí
$S_W$ se vuelve singular: tiene 12 autovalores nulos y un número de condición de casi $10^{18}$. Sin
regularizar, el método acierta 51 %, que es prácticamente como lanzar una moneda, y con la pseudoinversa
llega a 56 %. Pero si usamos $S_W + \lambda I$, la matriz ya se puede invertir y la exactitud sube hasta 63 %.
Lo curioso es que, con $\lambda$ grande, el resultado queda casi igual al de la diferencia de medias, que es
justo lo que dice la teoría: cuando $\lambda$ crece, $w$ tiende a $d$. El $\lambda$ lo elegimos con validación
cruzada, usando solo el entrenamiento, para no hacer trampa con los datos de prueba, y con eso llegamos a 62 %.

## Conclusiones y limitaciones (8:45 a 9:35)

**Quién:** Marcelo Torres  
**Pantalla:** Diapositiva de conclusiones.

En resumen, Fisher mejora la dirección de las medias porque toma en cuenta cómo están dispersas las clases,
y cuando hay pocos datos hace falta regularizar. Ahora, también tenemos que decir las limitaciones: usamos
datos sintéticos con la misma covarianza en las dos clases, que es justo el caso que le conviene a Fisher, y
con muy pocos datos Fisher termina pareciéndose a la diferencia de medias. Así que no podemos decir que Fisher
siempre sea el mejor.

## Cierre (9:35 a 10:00)

**Quién:** Marcelo Torres  
**Pantalla:** Referencias principales y enlace al repositorio.

Todas las referencias y el código están en nuestro repositorio de GitHub, el enlace está en pantalla.
Muchas gracias por vernos.
