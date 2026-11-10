# Caso aplicado: clasificación de riesgo de acciones del S&P 500

Aplicamos el discriminante de Fisher a datos reales: cada acción del S&P 500 (sin los sectores
Financials y Real Estate) es un vector x = (σ, β, MDD, log(1 + D/E)) ∈ ℝ⁴ calculado con datos del
1-nov-2022 al 1-nov-2025, y su etiqueta es 2 (riesgosa) si entre el 1-nov-2025 y el 1-nov-2026
su caída máxima superó 30 %, o 1 (no riesgosa) en otro caso.

Comparamos Fisher con PCA, la diferencia de medias, un indicador de pesos iguales y una regla de
persistencia, siempre con la misma regla de umbral.

## Cómo ejecutarlo

Los datos ya están guardados en `datos/`, así que **no hace falta internet**:

```bash
python aplicacion/correr_todo.py
```

Eso corre todos los análisis en orden y deja las tablas en `resultados/` y las figuras en
`figuras/`. Cada script también se puede correr por separado con F5 en Spyder.

Volver a descargar los datos de Yahoo Finance es opcional y tarda entre 5 y 40 minutos según la conexión
(`pip install yfinance`, luego `python aplicacion/datos_300.py`). Los números pueden cambiar un
poco porque Yahoo Finance corrige datos con el tiempo; los resultados del proyecto se obtuvieron
con los archivos guardados.

## Archivos

| Archivo | Qué hace |
|---|---|
| `comun.py` | Parámetros y funciones compartidas: carga, estandarización, direcciones (Fisher, PCA, diferencia de medias, pesos iguales), umbral, métricas y AUC |
| `datos_300.py` | Descarga los datos y arma las 300 acciones de entrenamiento (opcional) |
| `escenarios_lda_pca.py` | Validación interna: Fisher frente a PCA y la diferencia de medias en 20 particiones 70/30 |
| `validacion_externa.py` | Prueba única con acciones nuevas que el modelo nunca vio |
| `analisis_espectral.py` | Descomposición espectral de w = Σ⁻¹d y camino de regularización de γ = 0 (Fisher) a γ → ∞ (diferencia de medias) |
| `comparacion_sin_umbral.py` | AUC (compara solo la dirección, sin umbral), intervalos de Wilson y prueba de McNemar |
| `estabilidad_pesos.py` | Bootstrap: cuánto varían los pesos de w y si su signo es confiable |
| `tres_clases.py` | Fisher con tres clases de riesgo y el plano span(w₁, w₂) |
| `correr_todo.py` | Corre todo en orden a partir de los datos guardados |

| Datos (`datos/`) | Contenido |
|---|---|
| `acciones_dataset_300.csv` | Entrenamiento: 300 acciones con sus 4 variables, `mdd_futuro` y la etiqueta |
| `acciones_validacion.csv` | Validación externa: las acciones nuevas, guardadas para que la prueba no cambie |
| `sp500_lista.csv` | Copia de la lista del S&P 500 usada para elegir las acciones |
| `cola_usada.txt` | Cuántas acciones de la lista consumió el entrenamiento (la validación empieza después) |

## Convención de signo

En este caso usamos d = μ₂ − μ₁ y orientamos w para que un puntaje wᵀx alto signifique más riesgo.
Es la misma dirección de Fisher que en `src/fisher.py` (allí d = μ₁ − μ₂) con el signo opuesto, y
J(−w) = J(w). La prueba `tests/test_aplicacion.py` lo comprueba.

## Limitaciones

- **Sesgo de supervivencia:** usamos la lista actual del S&P 500, que excluye empresas que salieron
  del índice, probablemente más riesgosas.
- **Fecha del balance:** D/E usa el último balance con cierre anterior al 1-nov-2025, pero muchas
  empresas lo publican 2 o 3 meses después, así que no siempre era público en esa fecha.
- **Patrimonio negativo:** las empresas con patrimonio negativo se descartan porque su D/E no tiene
  sentido económico, y eso cambia la muestra.
- **Una sola etiqueta:** un solo año posterior y un solo umbral (30 %).
