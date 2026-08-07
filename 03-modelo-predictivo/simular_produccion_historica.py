"""
Simulación retroactiva de la lógica de producción — Personal OS / Investment Intelligence
==============================================================================
Este script responde a una pregunta concreta: si la lógica de semáforo +
zona muerta (±5%) + verificación a 5 días que armamos en n8n hubiera estado
corriendo TODOS LOS DÍAS durante los últimos ~3 años, ¿qué tasa de aciertos
real habría tenido el modelo?

Es la forma de obtener una estimación con cientos de observaciones AHORA,
en vez de esperar 6-12 meses a que se acumulen en producción. No reemplaza
al tracking en vivo — lo complementa: si ambos números terminan siendo
parecidos, da mucha más confianza en que la tasa de aciertos es real y no
un artefacto de la muestra chica.

Importante: usa el modelo YA ENTRENADO (los pesos actuales en producción),
no vuelve a entrenar nada. Esto es pura evaluación retrospectiva, con la
misma restricción de siempre: en el día i, solo se usan datos hasta el
día i para calcular las features — nunca se mira el futuro, excepto para
la única cosa que SÍ es al futuro a propósito: comparar contra el precio
real 5 días después, que es justamente lo que se está evaluando.

Requisitos: pip install pandas numpy requests scipy
Uso: python simular_produccion_historica.py
"""

import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

np.set_printoptions(precision=4, suppress=True)

# ============================================================
# CONFIGURACIÓN — debe coincidir EXACTO con lo que corre en n8n
# ============================================================
HORIZONTE = 5           # días hacia adelante que predice el modelo
UMBRAL_VERDE = 53       # probabilidad > esto -> verde
UMBRAL_ROJO = 47        # probabilidad < esto -> rojo (entre medio: amarillo)
ZONA_MUERTA_PCT = 3     # movimiento menor a esto (%) no cuenta como acierto/error, salvo amarillo

# Pesos y covarianza del modelo ACTUALMENTE en producción (misma fila del
# Google Sheet "Modelo Predictivo" que usa n8n) — actualizar acá también
# cada vez que se reentrene, para que esta simulación siga representando
# al modelo real.
PESOS_MODELO = np.array([0.0214, 0.04204, -0.05404, 0.04604, 0.03333, 0.00104, -0.00216,
                          -0.03295, -0.01254, 0.01045, 0.00762, -0.03692, -0.04895])
COVARIANZA_MODELO = np.array([
  [0.007853, -0.001765, -0.000488, -0.001137, -0.000588, -0.000889, -0.000791, -0.000637, -0.000932, -0.001945, -0.00086, -0.000843, -0.000219],
  [-0.001765, 0.007941, -0.000466, -0.000895, -0.000446, -0.000754, -0.000655, -0.000537, -0.00086, -0.001572, -0.000669, -0.000698, -0.000165],
  [-0.000488, -0.000466, 0.007777, -0.000539, 0.000282, -0.000884, -0.000718, -0.001512, -0.000384, -0.000326, -0.00023, -0.000242, -8e-06],
  [-0.001137, -0.000895, -0.000539, 0.0091, -0.00028, -0.000595, -0.000543, -0.000544, -0.000514, -0.001003, -0.000446, -0.000443, -7.2e-05],
  [-0.000588, -0.000446, 0.000282, -0.00028, 0.009294, -8.6e-05, -0.000286, 9e-05, -0.000212, -0.000633, -0.000238, -0.000234, -6.1e-05],
  [-0.000889, -0.000754, -0.000884, -0.000595, -8.6e-05, 0.009183, -0.00073, -0.000892, -0.000447, -0.000768, -0.000353, -0.000337, -7.9e-05],
  [-0.000791, -0.000655, -0.000718, -0.000543, -0.000286, -0.00073, 0.008854, -0.000936, -0.000232, -0.000736, -0.000318, -0.000311, -9.4e-05],
  [-0.000637, -0.000537, -0.001512, -0.000544, 9e-05, -0.000892, -0.000936, 0.008482, -0.000295, -0.000512, -0.000246, -0.000245, -1.7e-05],
  [-0.000932, -0.00086, -0.000384, -0.000514, -0.000212, -0.000447, -0.000232, -0.000295, 0.008607, -0.000825, -0.00033, -0.000395, 5.2e-05],
  [-0.001945, -0.001572, -0.000326, -0.001003, -0.000633, -0.000768, -0.000736, -0.000512, -0.000825, 0.008183, -0.000782, -0.000765, -0.000197],
  [-0.00086, -0.000669, -0.00023, -0.000446, -0.000238, -0.000353, -0.000318, -0.000246, -0.00033, -0.000782, 0.007558, -0.000339, -7.4e-05],
  [-0.000843, -0.000698, -0.000242, -0.000443, -0.000234, -0.000337, -0.000311, -0.000245, -0.000395, -0.000765, -0.000339, 0.007576, -3.5e-05],
  [-0.000219, -0.000165, -8e-06, -7.2e-05, -6.1e-05, -7.9e-05, -9.4e-05, -1.7e-05, 5.2e-05, -0.000197, -7.4e-05, -3.5e-05, 0.008391],
])
AÑOS_HISTORIAL = 3

# ============================================================
# 1. DESCARGAR HISTORIAL (idéntico al script de entrenamiento)
# ============================================================
def descargar_klines(symbol, años_historial):
    url = "https://api.binance.com/api/v3/klines"
    fin_dt = datetime.now(timezone.utc)
    inicio_dt = fin_dt - timedelta(days=365 * años_historial)
    fin_ms = int(fin_dt.timestamp() * 1000)
    cursor_ms = int(inicio_dt.timestamp() * 1000)
    klines = []
    while cursor_ms < fin_ms:
        params = {"symbol": symbol, "interval": "1d", "startTime": cursor_ms, "endTime": fin_ms, "limit": 1000}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        lote = resp.json()
        if not lote:
            break
        klines.extend(lote)
        cursor_ms = lote[-1][0] + 1
        if len(lote) < 1000:
            break
    fechas = [pd.to_datetime(k[0], unit="ms") for k in klines]
    precios = [float(k[4]) for k in klines]
    volumenes = [float(k[5]) for k in klines]
    df_symbol = pd.DataFrame({"fecha": fechas, "precio": precios, "volumen": volumenes}).set_index("fecha")
    return df_symbol[~df_symbol.index.duplicated(keep="first")].sort_index()

print(f"Descargando historial de ETH y BTC ({AÑOS_HISTORIAL} años, diario) desde Binance...")
df_eth = descargar_klines("ETHUSDT", AÑOS_HISTORIAL)
df_btc = descargar_klines("BTCUSDT", AÑOS_HISTORIAL)
df = df_eth.join(df_btc[["precio"]], rsuffix="_btc", how="inner")
print(f"Descargados {len(df)} días alineados ({df.index.min().date()} a {df.index.max().date()}).")


# ============================================================
# 2. LAS MISMAS 12 FEATURES DEL ENTRENAMIENTO
# ============================================================
def calcular_features(precios_arr, volumen_arr, precios_btc_arr, fecha, i):
    ventana = precios_arr[: i + 1]
    vol_ventana = volumen_arr[: i + 1]
    ventana_btc = precios_btc_arr[: i + 1]
    if len(ventana) < 91:
        return None
    retornos = np.diff(ventana) / ventana[:-1]
    precio_actual = ventana[-1]

    vol_reciente = np.std(retornos[-30:]); vol_base = np.std(retornos[-90:])
    ratio_vol = vol_reciente / vol_base if vol_base > 0 else 1
    score_vol = np.clip(100 - max(0, ratio_vol - 1) * 150, 0, 100)

    sma7 = np.mean(ventana[-7:]); sma30 = np.mean(ventana[-30:])
    señales = int(precio_actual > sma7) + int(sma7 > sma30) + int(precio_actual > sma30)
    score_tendencia = (señales / 3) * 100
    tendencia_alcista = precio_actual > sma30

    retornos90 = retornos[-90:]; mean_r = np.mean(retornos90)
    num = np.sum((retornos90[1:] - mean_r) * (retornos90[:-1] - mean_r))
    den = np.sum((retornos90 - mean_r) ** 2)
    autocorr = num / den if den > 0 else 0
    if autocorr >= 0:
        score_momentum = 50 + autocorr * 200 if tendencia_alcista else 50 - autocorr * 200
    else:
        score_momentum = 50 - autocorr * 100
    score_momentum = np.clip(score_momentum, 0, 100)

    ath_hasta_hoy = np.max(ventana)
    score_ath = np.clip((1 - precio_actual / ath_hasta_hoy) * 100, 0, 100)

    r14 = retornos[-14:]
    avg_gain = np.mean(np.clip(r14, 0, None)); avg_loss = np.mean(np.clip(-r14, 0, None))
    rs = avg_gain / avg_loss if avg_loss > 0 else np.inf
    score_rsi = 100 - 100 / (1 + rs) if np.isfinite(rs) else 100.0

    serie = pd.Series(ventana)
    ema12 = serie.ewm(span=12, adjust=False).mean(); ema26 = serie.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26; signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histograma = macd_line.iloc[-1] - signal_line.iloc[-1]
    score_macd = (np.tanh((histograma / precio_actual) * 50) + 1) / 2 * 100

    sma20 = np.mean(ventana[-20:]); std20 = np.std(ventana[-20:])
    banda_sup = sma20 + 2*std20; banda_inf = sma20 - 2*std20
    pct_b = (precio_actual - banda_inf) / (banda_sup - banda_inf) if banda_sup > banda_inf else 0.5
    score_bollinger = np.clip(pct_b, 0, 1) * 100

    vol7 = np.mean(vol_ventana[-7:]); vol30 = np.mean(vol_ventana[-30:])
    ratio_volumen = vol7 / vol30 if vol30 > 0 else 1
    score_volumen = (np.tanh((ratio_volumen - 1) * 2) + 1) / 2 * 100

    retornos_btc = np.diff(ventana_btc) / ventana_btc[:-1]
    n_corr = min(30, len(retornos), len(retornos_btc))
    if n_corr >= 5:
        corr = np.corrcoef(retornos[-n_corr:], retornos_btc[-n_corr:])[0, 1]
        corr = 0.0 if np.isnan(corr) else corr
    else:
        corr = 0.0
    score_corr_btc = (corr + 1) / 2 * 100

    dow = fecha.dayofweek
    score_dow_sin = (np.sin(2*np.pi*dow/7) + 1) / 2 * 100
    score_dow_cos = (np.cos(2*np.pi*dow/7) + 1) / 2 * 100
    ultimo_dia_mes = (fecha + pd.offsets.MonthEnd(0)).day
    score_fin_mes = 100.0 if (ultimo_dia_mes - fecha.day) < 3 else 0.0

    return np.array([
        score_vol, score_tendencia, score_momentum, score_ath,
        score_rsi, score_macd, score_bollinger, score_volumen, score_corr_btc,
        score_dow_sin, score_dow_cos, score_fin_mes,
    ]) / 100.0


def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-np.clip(a, -500, 500)))

def predictive_prob_bayes(phi, w, S_N):
    mu_a = phi @ w
    sigma_a2 = phi @ S_N @ phi
    kappa = 1.0 / np.sqrt(1.0 + np.pi * sigma_a2 / 8.0)
    return sigmoid(kappa * mu_a)


# ============================================================
# 3. SIMULAR día por día — misma lógica exacta que corre en n8n
# ============================================================
precios_arr = df["precio"].values
volumen_arr = df["volumen"].values
precios_btc_arr = df["precio_btc"].values
fechas_arr = df.index

resultados = []
for i in range(91, len(precios_arr) - HORIZONTE):
    feats = calcular_features(precios_arr, volumen_arr, precios_btc_arr, fechas_arr[i], i)
    if feats is None:
        continue
    phi = np.concatenate([[1.0], feats])
    prob = predictive_prob_bayes(phi, PESOS_MODELO, COVARIANZA_MODELO) * 100

    color = "verde" if prob > UMBRAL_VERDE else "rojo" if prob < UMBRAL_ROJO else "amarillo"

    precio_hoy = precios_arr[i]
    precio_futuro = precios_arr[i + HORIZONTE]
    cambio_pct = (precio_futuro - precio_hoy) / precio_hoy * 100
    subio = cambio_pct > 0

    if color == "amarillo":
        resultado = "acierto" if abs(cambio_pct) < ZONA_MUERTA_PCT else "error"
    elif abs(cambio_pct) < ZONA_MUERTA_PCT:
        resultado = "movimiento_insignificante"
    elif color == "verde":
        resultado = "acierto" if subio else "error"
    else:
        resultado = "error" if subio else "acierto"

    resultados.append({"fecha": fechas_arr[i], "prob": prob, "color": color,
                        "cambio_pct": cambio_pct, "resultado": resultado})

res_df = pd.DataFrame(resultados)
print(f"\nSimulación completa: {len(res_df)} días evaluados "
      f"({res_df['fecha'].min().date()} a {res_df['fecha'].max().date()})")


# ============================================================
# 4. RESULTADOS — tasa simple + Bayesiana (Beta-Binomial) + desglose por color
# ============================================================
evaluables = res_df[res_df["resultado"].isin(["acierto", "error"])]
aciertos = (evaluables["resultado"] == "acierto").sum()
errores = (evaluables["resultado"] == "error").sum()
n_eval = len(evaluables)

print("\n===== RESULTADO GLOBAL =====")
print(f"Evaluables (acierto/error): {n_eval}  |  Movimiento insignificante: "
      f"{(res_df['resultado']=='movimiento_insignificante').sum()}")
print(f"Aciertos: {aciertos}  |  Errores: {errores}")

if n_eval > 0:
    tasa_simple = aciertos / n_eval
    alpha_post = 1 + aciertos
    beta_post = 1 + errores
    media_bayes = alpha_post / (alpha_post + beta_post)
    from scipy import stats
    ci_low, ci_high = stats.beta.ppf([0.025, 0.975], alpha_post, beta_post)

    print(f"\nTasa simple (aciertos/total): {tasa_simple:.1%}")
    print(f"Media Bayesiana (Beta({alpha_post},{beta_post})): {media_bayes:.1%}")
    print(f"Intervalo de credibilidad 95%: [{ci_low:.1%}, {ci_high:.1%}]")

print("\n===== DESGLOSE POR COLOR =====")
for color in ["verde", "amarillo", "rojo"]:
    sub = res_df[res_df["color"] == color]
    sub_eval = sub[sub["resultado"].isin(["acierto", "error"])]
    if len(sub_eval) == 0:
        print(f"{color:>10}: sin datos evaluables ({len(sub)} predicciones totales)")
        continue
    tasa = (sub_eval["resultado"] == "acierto").mean()
    print(f"{color:>10}: {tasa:.1%} de aciertos sobre {len(sub_eval)} evaluables "
          f"({len(sub)} predicciones totales)")

print(
    "\nNota: estas verificaciones se solapan (ventanas de 5 días superpuestas día a día),\n"
    "así que el tamaño de muestra EFECTIVO (independiente) es menor al n mostrado —\n"
    "aproximadamente n/5. Tratá el intervalo de credibilidad de arriba como optimista."
)
