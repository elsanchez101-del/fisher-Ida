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

**Duración total:** unos 10 minutos (lo permitido es de 9 a 11 minutos). Cada integrante habla unos 2 min 35 s y expone tres diapositivas. El texto está escrito tal como se dice en voz alta: las fórmulas ya están "traducidas" a palabras, así que solo hay que leerlo.

| Integrante | Diapositivas | Tiempo |
|---|---|---|
| Elison Sanchez | 1 a 3 | 0:00 a 2:35 |
| Marcelo Torres | 4 a 6 | 2:35 a 5:10 |
| Marcelo Bravo | 7 a 9 | 5:10 a 7:50 |
| Rodolfo Brandariz | 10 a 12 | 7:50 a 10:30 |

# Elison Sanchez (0:00 a 2:35)

## Diapositiva 1 · Título (0:00 a 0:20)

"Hola, qué tal. Somos Elison Sanchez, Rodolfo Brandariz, Marcelo Bravo y Marcelo Torres, del curso de Álgebra
Lineal Aplicada. Les presentamos nuestro proyecto: un modelo para predecir el riesgo de una acción usando el
discriminante lineal de Fisher. Yo soy Elison y les voy a contar el problema."

## Diapositiva 2 · Problema, objetivo y alcance (0:20 a 1:30)

"Cada acción tiene varias medidas, y queremos resumirlas en un solo número que diga si es
riesgosa o no. Ese número se obtiene multiplicando cada medida por un peso y sumando todo; en álgebra lineal
eso es un producto interno. Geométricamente, es proyectar los datos sobre una recta, como la sombra de un
objeto sobre una pared. La pregunta es qué recta elegir para que las dos clases queden separadas. Lo más obvio
sería unir los promedios de cada grupo, pero cuando las variables están relacionadas, las nubes son alargadas
y las sombras se enciman.

Nuestro objetivo fue programar el discriminante de Fisher, comprobarlo con un ejemplo a mano y con la
librería scikit-learn, ver cuándo le gana a otros métodos más simples, qué hacer cuando una matriz no se puede
invertir, y aplicarlo a acciones reales. El alcance es el del artículo de referencia: dos clases con la misma
forma. Para todo esto usamos producto interno, formas cuadráticas, el teorema espectral, la desigualdad de
Cauchy-Schwarz, sistemas de ecuaciones y autovalores."

## Diapositiva 3 · Riesgo financiero y datos (1:30 a 2:35)

"En finanzas, el riesgo es qué tanto se dispersa el retorno. Miren estos dos activos: los dos rinden más o
menos seis por ciento, pero el activo A tiene una desviación estándar de diecisiete coma siete por ciento, y el
B, de solo dos coma ocho. El A es mucho más riesgoso.

Por eso, para cada acción del S&P 500 calculamos cuatro medidas con los precios de noviembre de 2022 a noviembre
de 2025: la volatilidad, la beta, que dice cuánto se mueve con el mercado, la caída máxima que tuvo y su nivel
de deuda. Esas cuatro forman un vector. La etiqueta viene del futuro: una acción es
riesgosa si en el año siguiente cayó más de treinta por ciento. Entrenamos con trescientas acciones y dejamos
cuarenta nuevas para la prueba final. Ahora Marcelo les explica cómo funciona Fisher."

# Marcelo Torres (2:35 a 5:10)

## Diapositiva 4 · El criterio de Fisher (2:35 a 3:25)

**En pantalla:** mover el deslizador despacio y al final pulsar "Fisher".

"Gracias, Elison. Soy Marcelo Torres. La idea de Fisher es mirar dos cosas a la vez: que los promedios de las
dos clases queden lejos cuando los proyectamos, y que cada clase quede bien apretada. Por eso define un
cociente, J. Arriba va la distancia entre los promedios proyectados, al cuadrado. Abajo, qué tan dispersas
quedan las sombras dentro de cada clase, que se calcula con la matriz de dispersión, S sub W.

Si giro la recta, cambia J, y el punto más alto está justo en la dirección de Fisher. Además, si multiplico el
vector de pesos por cualquier número, J no cambia: solo importa la dirección."

## Diapositiva 5 · Demostración (3:25 a 4:30)

**En pantalla:** pulsar los pasos 1, 2, 3 y 4 en orden mientras se habla.

"¿Por qué esa es la mejor dirección? Lo demostramos en cuatro pasos.

Uno: S sub W es simétrica y sus autovalores son positivos, así que, por el teorema espectral, tiene raíz
cuadrada.

Dos: hacemos un cambio de variable. Llamamos v a la raíz de S sub W por el vector de pesos. Así, la parte de
abajo del cociente es solo el largo de v al cuadrado, y las nubes se vuelven redondas.

Tres: por la desigualdad de Cauchy-Schwarz, la sombra de un vector nunca es más larga que el vector. Eso le pone
un techo a J, y el techo se alcanza cuando v apunta hacia la recta que une los dos centros.

Cuatro: deshacemos el cambio, y la mejor dirección queda como la inversa de S sub W por la diferencia de los
promedios. Si las nubes ya fueran redondas, Fisher daría lo mismo que unir los promedios."

## Diapositiva 6 · Sistema lineal y ejemplo a mano (4:30 a 5:10)

"En la práctica, esa dirección se obtiene resolviendo un sistema de ecuaciones, sin invertir la matriz. Lo
comprobamos a mano con tres puntos por clase. La matriz de
dispersión nos dio cuatro tercios en la diagonal y dos tercios fuera de ella, y al resolver el sistema la
dirección salió proporcional a cuatro, uno. Al proyectar, la primera clase cae en cinco, nueve y diez, y la
segunda en diecinueve, veintitrés y veinticuatro. El umbral, que es el punto medio entre los promedios, es
quince, y el valor máximo de J es siete. El código da exactamente lo mismo. Le paso a Marcelo."

# Marcelo Bravo (5:10 a 7:50)

## Diapositiva 7 · Caso A y verificaciones (5:10 a 6:20)

"Gracias. Soy Marcelo Bravo. Primero probamos con datos que generamos nosotros: dos grupos alargados e
inclinados, repitiendo diez veces. Fisher acertó sesenta y nueve
coma cinco por ciento; la recta que une los promedios, cincuenta y siete coma nueve; y PCA, cincuenta y cinco
coma nueve. PCA pierde porque busca la dirección donde los datos varían más, que es la diagonal larga de la
nube, y justo ahí los grupos se mezclan. Como conocíamos la distribución verdadera, también calculamos lo
máximo que se podía acertar: setenta y uno coma siete por ciento. Fisher se quedó a solo dos puntos.

¿Cómo sabemos que el código está bien? Reproduce el ejemplo a mano, da la misma dirección que scikit-learn,
ninguna dirección al azar le gana, y tiene veinte pruebas automáticas.
Aquí ven el comando que vuelve a correr todo y termina diciendo 'todo ok'."

## Diapositiva 8 · Caso B: regularización (6:20 a 7:05)

"¿Y si hay pocos datos? Probamos con cuarenta variables y solo treinta datos. Así, la matriz S sub W tiene doce
autovalores iguales a cero y no se puede invertir. Sin arreglarlo, Fisher
acierta cincuenta y uno por ciento, que es como lanzar una moneda. La solución es sumarle un número pequeño,
lambda, a toda la diagonal de la matriz. Eso le suma lambda a todos los autovalores, y ya ninguno queda en
cero. Eligiendo ese número con validación cruzada, usando solo los datos de entrenamiento, la exactitud sube a
sesenta y uno coma nueve por ciento. Y si lambda es muy grande, Fisher termina igual que la recta de los
promedios."

## Diapositiva 9 · Resultados con acciones (7:05 a 7:50)

"Ahora el caso real. En la validación interna, separando las trescientas acciones en entrenamiento y prueba
veinte veces, Fisher acertó sesenta y nueve coma seis por ciento, contra cincuenta y siete coma ocho de decir
siempre 'no riesgosa'. En la prueba final con cuarenta acciones que el modelo nunca vio, los cinco métodos
acertaron veinticinco de cuarenta, y Fisher detectó dieciséis de las veintitrés riesgosas. O sea, funciona, pero
no le gana a métodos más simples, y con cuarenta acciones no hay un ganador claro.
Rodolfo les explica por qué."

# Rodolfo Brandariz (7:50 a 10:30)

## Diapositiva 10 · Pesos y tres clases (7:50 a 8:35)

"Gracias. Soy Rodolfo Brandariz. El vector de pesos nos dice cuánto importa cada variable: la volatilidad, con
cero coma setenta y dos, y la beta, con cero coma sesenta y uno, son las que más pesan, y la deuda casi no
aporta. Con tres niveles de
riesgo, Fisher da como máximo dos direcciones, que forman un plano. Ahí el riesgo alto se separa, pero el bajo y el medio se mezclan."

## Diapositiva 11 · Por qué Fisher no gana (8:35 a 9:45)

**En pantalla:** cambiar entre "Caso A" y "Acciones".

"La explicación sale del teorema espectral. La nube de datos tiene ejes, que son sus autovectores, y cada eje
tiene un ancho, que es su autovalor. Si escribimos la dirección de Fisher usando esos ejes, vemos que Fisher
divide entre el ancho de cada eje. O sea, le da más peso a los ejes angostos.

En el caso A, parte de la separación estaba en un eje angosto, y por eso Fisher ganaba por doce puntos. Pero en
las acciones, casi toda la separación cae en el primer eje, que es también el más ancho: una especie de
'riesgo general', donde volatilidad, beta y caída suben juntas. El noventa y tres por ciento del vector de
Fisher está en ese eje, y por eso termina apuntando casi igual que la recta de los promedios, a solo once
grados. Y si regularizamos, el vector gira hasta coincidir con ella y la exactitud no cambia."

## Diapositiva 12 · Conclusiones, limitaciones y referencias (9:45 a 10:30)

"En resumen: demostramos que la dirección de Fisher es la que maximiza su criterio. Cuando la separación está
en una dirección angosta, Fisher le gana a PCA y a la recta de los promedios, y cuando hay pocos datos hay que
regularizar. En las acciones empata, porque la separación está en la dirección más ancha. Las limitaciones son
que usamos la lista actual del índice, que deja fuera a las empresas que salieron, un solo año, un solo umbral
de caída, y una prueba de solo cuarenta acciones. Nos basamos en el artículo de Fisher, los libros de Hastie
y de Bishop, scikit-learn y el material del curso. Todo está en nuestro repositorio de GitHub, que ven en
pantalla. Muchas gracias."

# Recomendaciones para grabar

- Cada uno graba su parte con su diapositiva en pantalla; en las diapositivas 4, 5 y 11 hagan los movimientos indicados.
- Lean el guion varias veces antes de grabar para que suene natural, no leído.
- Si al unir las partes pasa de 11 minutos, recorten un poco las diapositivas 3 y 10; si queda por debajo de 9, hablen con más calma en las diapositivas 5 y 11.
- Comprueben que el enlace del video se pueda ver sin pedir permiso antes de entregarlo.
