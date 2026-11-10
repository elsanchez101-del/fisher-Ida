# -*- coding: utf-8 -*-
"""
datos_300.py
------------
Construye el conjunto de datos de 300 acciones del S&P 500 (sin financieras ni bienes raíces).
SOLO hace falta correrlo si se quieren volver a descargar los datos: el repositorio ya trae
datos/acciones_dataset_300.csv, y los demás scripts leen ese archivo.

VECTOR x de cada acción:
    sigma : volatilidad anual      = desv. estándar de retornos diarios * raiz(252)
    beta  : beta frente al S&P 500 = Cov(r_accion, r_mercado) / Var(r_mercado)
    mdd   : caída máxima (max drawdown) = mayor caída desde un máximo previo
    de    : apalancamiento = Total Debt / Stockholders Equity (último balance anual <= FIN_VARS)

ETIQUETA y (se calcula con el año SIGUIENTE a la ventana de variables):
    y = 2 (riesgosa)    si el mdd de ese año supera UMBRAL
    y = 1 (no riesgosa) en otro caso

SELECCIÓN DE ACCIONES (no depende de la etiqueta):
    cola = lista inicial fija (INICIALES) + resto del S&P 500 en orden aleatorio con semilla fija.
    Se procesa de 10 en 10 hasta tener N_OBJETIVO acciones válidas.
    La lista del S&P 500 se guarda en datos/sp500_lista.csv la primera vez y después se lee de
    ahí: la lista publicada cambia con el tiempo y con ella cambiaría la cola de acciones.

USO: F5 en Spyder o `python aplicacion/datos_300.py` (tarda unos 35-45 minutos).
Requiere: pip install yfinance
"""
import os
import time
import numpy as np
import pandas as pd

import comun as c

# ----------------------------- PARÁMETROS ---------------------------------
INICIO     = "2022-11-01"   # inicio de la ventana de variables
FIN_VARS   = "2025-11-01"   # fin de la ventana de variables / inicio del año de la etiqueta
FIN_ETIQ   = "2026-11-01"   # fin del año de la etiqueta
MERCADO    = "^GSPC"
N_OBJETIVO = 300
LOTE       = 10
URL_LISTA  = ("https://raw.githubusercontent.com/datasets/"
              "s-and-p-500-companies/main/data/constituents.csv")
EXCLUIR_SECTORES = ["Financials", "Real Estate"]

# Lista inicial (90 acciones, sin financieras). Se procesan primero, en este orden.
INICIALES = [
    "AAPL", "MSFT", "NVDA", "ORCL", "CSCO", "ADBE", "TXN", "QCOM", "INTC", "AMD",
    "IBM", "MU", "AMAT", "INTU", "CRM", "ACN",
    "GOOGL", "META", "NFLX", "DIS", "VZ", "T", "CMCSA", "TMUS",
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TGT", "F", "GM", "BKNG", "TJX",
    "PG", "KO", "PEP", "WMT", "COST", "MDLZ", "CL", "KHC", "MO", "PM", "ADM",
    "JNJ", "PFE", "MRK", "ABBV", "LLY", "UNH", "AMGN", "BMY", "GILD", "MDT",
    "CVS", "TMO", "ABT", "DHR",
    "CAT", "BA", "HON", "UPS", "GE", "LMT", "DE", "MMM", "RTX", "FDX", "UNP", "WM", "EMR",
    "XOM", "CVX", "COP", "SLB", "OXY", "KMI", "EOG",
    "NEE", "DUK", "SO", "D", "LIN", "DOW", "NEM", "FCX", "APD",
]


# ------------------------- FUNCIONES DE CÁLCULO ----------------------------
def max_drawdown(precios):
    """Para cada columna: max sobre t de (máximo acumulado - precio) / máximo acumulado."""
    maximo = precios.cummax()
    return ((maximo - precios) / maximo).max()


def volatilidad_anual(retornos):
    return retornos.std() * np.sqrt(252)


def beta(retornos, ret_mercado):
    """Pendiente de la regresión de cada acción sobre el mercado."""
    out = {}
    for t in retornos.columns:
        par = pd.concat([retornos[t], ret_mercado], axis=1).dropna()
        out[t] = par.iloc[:, 0].cov(par.iloc[:, 1]) / par.iloc[:, 1].var()
    return pd.Series(out)


def construir_dataset(precios, de):
    """precios: cierres ajustados (fechas x tickers, incluye MERCADO). de: Series de apalancamiento."""
    pre  = precios[(precios.index >= INICIO) & (precios.index < FIN_VARS)]
    post = precios[(precios.index >= FIN_VARS) & (precios.index < FIN_ETIQ)]
    acciones = [t for t in precios.columns if t != MERCADO]
    pre_a, post_a = pre[acciones], post[acciones]
    ret, ret_m = pre_a.pct_change(), pre[MERCADO].pct_change()
    df = pd.DataFrame({
        "sigma": volatilidad_anual(ret),
        "beta": beta(ret, ret_m),
        "mdd": max_drawdown(pre_a),
        "de": de.reindex(acciones),
        "mdd_futuro": max_drawdown(post_a),
        "dias_pre": pre_a.count(),
        "dias_post": post_a.count(),
    })
    df.index.name = "ticker"
    df["etiqueta"] = np.where(df["mdd_futuro"] > c.UMBRAL, 2, 1)
    return df


def filtrar(df):
    """Descarta acciones sin historia completa o con apalancamiento no válido."""
    sin_hist = (df["dias_pre"] < 700) | (df["dias_post"] < 200)
    sin_de = df["de"].isna()
    print(f"  descartadas por historia incompleta: {int(sin_hist.sum())}; "
          f"por D/E no válido: {int((sin_de & ~sin_hist).sum())}")
    return df[~sin_hist & ~sin_de].copy()


# --------------------------- FUNCIONES DE DESCARGA --------------------------
def apalancamiento(ticker):
    """Total Debt / Stockholders Equity del último balance anual con fecha <= FIN_VARS.
    Limitación: la fecha es la de cierre del ejercicio; el balance suele publicarse 2 o 3 meses
    después, así que en algunos casos aún no era público el 1-nov-2025."""
    import yfinance as yf
    try:
        bs = yf.Ticker(ticker).balance_sheet
        fechas = [f for f in bs.columns if pd.Timestamp(f).tz_localize(None) <= pd.Timestamp(FIN_VARS)]
        if not fechas:
            return np.nan
        col = max(fechas)
        deuda, patrimonio = bs.loc["Total Debt", col], bs.loc["Stockholders Equity", col]
        if pd.isna(deuda) or pd.isna(patrimonio) or patrimonio <= 0:
            return np.nan          # patrimonio negativo: D/E sin sentido económico, se descarta
        return float(deuda / patrimonio)
    except Exception as e:
        print(f"    {ticker}: sin balance ({e})")
        return np.nan


def procesar_lote(tickers):
    """Descarga y calcula variables y etiqueta de un lote. Si la descarga falla, devuelve vacío
    y el programa sigue con el siguiente lote."""
    import yfinance as yf
    try:
        datos = yf.download(tickers + [MERCADO], start=INICIO, end=FIN_ETIQ,
                            auto_adjust=True, progress=False)
        precios = datos["Close"]
        presentes = [t for t in tickers if t in precios.columns and precios[t].notna().any()]
        de = pd.Series({t: apalancamiento(t) for t in presentes}, dtype=float)
        time.sleep(0.5)
        return filtrar(construir_dataset(precios[presentes + [MERCADO]], de))
    except Exception as e:
        print(f"  ERROR en el lote {tickers}: {e}. Se omite.")
        return pd.DataFrame()


def lista_sp500():
    """Lista del S&P 500. La primera vez se descarga y se guarda; después se lee la copia."""
    if os.path.exists(c.ARCHIVO_LISTA):
        return pd.read_csv(c.ARCHIVO_LISTA)
    lista = pd.read_csv(URL_LISTA)
    c.asegurar_carpetas()
    lista.to_csv(c.ARCHIVO_LISTA, index=False)
    print(f"Lista del S&P 500 guardada en {c.ARCHIVO_LISTA}")
    return lista


def cola_de_acciones():
    """Lista inicial + resto del S&P 500 (sin sectores excluidos) en orden aleatorio reproducible."""
    lista = lista_sp500()
    lista = lista[~lista["GICS Sector"].isin(EXCLUIR_SECTORES)].copy()
    lista["Yahoo"] = lista["Symbol"].str.replace(".", "-", regex=False)   # BRK.B -> BRK-B
    resto = lista[~lista["Yahoo"].isin(INICIALES)]
    orden = np.random.default_rng(c.SEMILLA).permutation(len(resto))
    return INICIALES + resto.iloc[orden]["Yahoo"].tolist()


def descargar_cola(cola, objetivo, etiqueta_lote="Lote"):
    """Procesa la cola de 10 en 10 hasta juntar `objetivo` acciones válidas.
    Devuelve (DataFrame, cuántos tickers de la cola se consumieron)."""
    partes, total, i = [], 0, 0
    while total < objetivo and i < len(cola):
        lote = cola[i:i + LOTE]
        i += LOTE
        print(f"\n{etiqueta_lote} {i // LOTE}: {lote}")
        df = procesar_lote(lote)
        if len(df):
            partes.append(df)
            total += len(df)
        print(f"  total acumulado: {total}")
    out = pd.concat(partes).iloc[:objetivo] if partes else pd.DataFrame()
    return out, i


# ------------------------------- PRINCIPAL ----------------------------------
def main():
    import yfinance as yf
    print("yfinance", yf.__version__)
    c.asegurar_carpetas()
    cola = cola_de_acciones()
    print(f"Acciones en cola: {len(cola)} (las primeras {len(INICIALES)} son la lista inicial)")

    final, usados = descargar_cola(cola, N_OBJETIVO)
    final.to_csv(c.ARCHIVO)
    with open(c.ARCHIVO_COLA, "w") as f:
        f.write(str(usados))     # la validación externa empieza DESPUÉS de este punto de la cola
    print(f"\nGuardado: {c.ARCHIVO} ({len(final)} acciones)")

    print("\n--- Balance de clases (1 = no riesgosa, 2 = riesgosa) ---")
    print(final["etiqueta"].value_counts().sort_index())
    print("\n--- Resumen de las 4 variables ---")
    print(final[c.VARS].describe().round(3))
    print("\n--- Correlación ---")
    print(final[c.VARS].corr().round(2))
    return final


if __name__ == "__main__":
    df = main()
