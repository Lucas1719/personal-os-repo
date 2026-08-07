"""
Comparación de horizontes de predicción — Personal OS / Investment Intelligence
==============================================================================
El horizonte de 5 días que usa el modelo en producción fue una elección
arbitraria al armar el script original, no el resultado de comparar
alternativas. Este script entrena el MISMO modelo (regresión logística
bayesiana, IRLS + Laplace, Bishop cap. 4.5) con 3 horizontes distintos —
3, 5 y 7 días — usando la MISMA metodología para cada uno (CV temporal
para elegir la regularización, hold-out final nunca visto durante el
ajuste), y los compara de forma justa.

Importante: esto NO prueba automáticamente que "el mejor horizonte medido
acá" sea el mejor de verdad — con datasets de este tamaño (~1000 días),
una diferencia de pocos puntos de log-loss entre horizontes puede ser
ruido de muestra, no una diferencia real. El script lo señala explícitamente
al final si las diferencias son chicas.

Requisitos: pip install pandas numpy requests scikit-learn
Uso: python comparar_horizontes.py
"""

import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

np.set_printoptions(precision=4, suppress=True)

AÑOS_HISTORIAL = 3
HORIZONTES_A_PROBAR = [3, 5, 7]

# ============================================================
# 1. DESCARGAR HISTORIAL — una sola vez, se reusa para los 3 horizontes
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
print(f"Descargados {len(df)} días alineados ({df.index.min().date()} a {df.index.max().date()}).\n")


# ============================================================
# 2. LAS MISMAS 12 FEATURES (no dependen del horizonte, solo el target sí)
# ============================================================
FEATURE_NAMES = [
    "volatilidad", "tendencia", "momentum", "ath",
    "rsi", "macd", "bollinger", "volumen", "corr_btc",
    "estacionalidad_sin", "estacionalidad_cos", "fin_de_mes",
]

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


# ============================================================
# 3. REGRESIÓN LOGÍSTICA BAYESIANA — misma implementación de siempre
# ============================================================
def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-np.clip(a, -500, 500)))

def fit_map_irls(Phi, t, alpha, max_iter=100, tol=1e-6):
    N, M = Phi.shape
    w = np.zeros(M)
    for _ in range(max_iter):
        a = Phi @ w
        y = sigmoid(a)
        R = np.diag(y * (1 - y) + 1e-9)
        grad = Phi.T @ (y - t) + alpha * w
        H = Phi.T @ R @ Phi + alpha * np.eye(M)
        w_new = w - np.linalg.solve(H, grad)
        if np.linalg.norm(w_new - w) < tol:
            w = w_new
            break
        w = w_new
    a = Phi @ w
    y = sigmoid(a)
    R = np.diag(y * (1 - y) + 1e-9)
    H = Phi.T @ R @ Phi + alpha * np.eye(M)
    S_N = np.linalg.inv(H)
    return w, S_N

def predictive_prob_bayes(phi, w, S_N):
    mu_a = phi @ w
    sigma_a2 = phi @ S_N @ phi
    kappa = 1.0 / np.sqrt(1.0 + np.pi * sigma_a2 / 8.0)
    return sigmoid(kappa * mu_a), sigma_a2

def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


# ============================================================
# 4. ENTRENAR Y EVALUAR UN HORIZONTE — misma metodología para los 3
# ============================================================
def evaluar_horizonte(horizonte, precios_arr, volumen_arr, precios_btc_arr, fechas_arr):
    filas_X, filas_y = [], []
    for i in range(91, len(precios_arr) - horizonte):
        feats = calcular_features(precios_arr, volumen_arr, precios_btc_arr, fechas_arr[i], i)
        if feats is None:
            continue
        subio = 1 if precios_arr[i + horizonte] > precios_arr[i] else 0
        filas_X.append(feats)
        filas_y.append(subio)

    X_all = np.array(filas_X)
    y_all = np.array(filas_y)
    n = len(X_all)

    split_final = int(n * 0.85)
    X_trainval, y_trainval = X_all[:split_final], y_all[:split_final]
    X_test, y_test = X_all[split_final:], y_all[split_final:]

    alphas_candidatos = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]
    tscv = TimeSeriesSplit(n_splits=5)

    mejor_alpha, mejor_logloss = None, np.inf
    for alpha in alphas_candidatos:
        losses = []
        for tr_idx, val_idx in tscv.split(X_trainval):
            Phi_tr = add_bias(X_trainval[tr_idx])
            Phi_val = add_bias(X_trainval[val_idx])
            w, S_N = fit_map_irls(Phi_tr, y_trainval[tr_idx], alpha)
            probs = np.array([predictive_prob_bayes(phi, w, S_N)[0] for phi in Phi_val])
            probs = np.clip(probs, 1e-6, 1 - 1e-6)
            losses.append(log_loss(y_trainval[val_idx], probs, labels=[0, 1]))
        avg_loss = np.mean(losses)
        if avg_loss < mejor_logloss:
            mejor_logloss, mejor_alpha = avg_loss, alpha

    Phi_trainval = add_bias(X_trainval)
    Phi_test = add_bias(X_test)
    w_final, S_N_final = fit_map_irls(Phi_trainval, y_trainval, mejor_alpha)

    probs_bayes = np.array([predictive_prob_bayes(phi, w_final, S_N_final)[0] for phi in Phi_test])
    probs_bayes_clip = np.clip(probs_bayes, 1e-6, 1 - 1e-6)
    pred_bayes = (probs_bayes >= 0.5).astype(int)
    baseline = max(y_test.mean(), 1 - y_test.mean())

    return {
        "horizonte": horizonte,
        "n_total": n,
        "n_test": len(X_test),
        "alpha_elegido": mejor_alpha,
        "accuracy": accuracy_score(y_test, pred_bayes),
        "baseline": baseline,
        "logloss": log_loss(y_test, probs_bayes_clip, labels=[0, 1]),
        "w_final": w_final,
        "S_N_final": S_N_final,
    }


# ============================================================
# 5. CORRER LOS 3 HORIZONTES Y COMPARAR
# ============================================================
precios_arr = df["precio"].values
volumen_arr = df["volumen"].values
precios_btc_arr = df["precio_btc"].values
fechas_arr = df.index

resultados = []
for h in HORIZONTES_A_PROBAR:
    print(f"Entrenando y evaluando horizonte = {h} días...")
    r = evaluar_horizonte(h, precios_arr, volumen_arr, precios_btc_arr, fechas_arr)
    resultados.append(r)
    print(f"  -> accuracy={r['accuracy']:.1%}  baseline={r['baseline']:.1%}  "
          f"logloss={r['logloss']:.4f}  alpha={r['alpha_elegido']}  n_test={r['n_test']}")

print("\n===== COMPARACIÓN FINAL =====")
print(f"{'Horizonte':>10} {'Accuracy':>10} {'Baseline':>10} {'Diff acc.':>10} {'Log-loss':>10} {'n_test':>8}")
for r in resultados:
    diff = (r["accuracy"] - r["baseline"]) * 100
    print(f"{r['horizonte']:>9}d {r['accuracy']:>9.1%} {r['baseline']:>9.1%} "
          f"{diff:>+9.1f}p {r['logloss']:>10.4f} {r['n_test']:>8}")

mejor = min(resultados, key=lambda r: r["logloss"])
peor = max(resultados, key=lambda r: r["logloss"])
diferencia_logloss = peor["logloss"] - mejor["logloss"]

print(f"\nMenor log-loss: horizonte de {mejor['horizonte']} días ({mejor['logloss']:.4f})")

if diferencia_logloss < 0.02:
    print(
        f"\n⚠️  La diferencia entre el mejor y el peor horizonte es de solo {diferencia_logloss:.4f}\n"
        "    en log-loss — con un hold-out de este tamaño, una diferencia así de chica está\n"
        "    dentro del margen de ruido esperable. No hay evidencia sólida de que un horizonte\n"
        "    sea realmente mejor que los otros; elegir uno u otro es prácticamente indistinto\n"
        "    con los datos actuales."
    )
else:
    print(
        f"\n✓ La diferencia de {diferencia_logloss:.4f} en log-loss es más notoria — sugiere\n"
        f"  que el horizonte de {mejor['horizonte']} días podría tener una ventaja real, aunque\n"
        "  conviene confirmarlo con más datos antes de cambiar el horizonte en producción."
    )

# ============================================================
# 6. EXPORTAR pesos del horizonte con menor log-loss, por si se decide adoptarlo
# ============================================================
print(f"\n===== PESOS DEL MEJOR HORIZONTE ({mejor['horizonte']} días) — por si se decide adoptarlo =====")
pesos_json = list(np.round(mejor["w_final"], 5).tolist())
covarianza_json = [list(np.round(row, 6).tolist()) for row in mejor["S_N_final"]]
print(f"""
// Horizonte: {mejor['horizonte']} días
const pesosModelo = {pesos_json};
const covarianzaModelo = [
{chr(10).join('  ' + str(row) + ',' for row in covarianza_json)}
];
const accuracyModelo = {mejor['accuracy']:.4f};
const logLossModelo = {mejor['logloss']:.4f};
const baselineModelo = {mejor['baseline']:.4f};
""")
