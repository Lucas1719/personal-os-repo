"""
Entrenador de modelo predictivo real — Personal OS / Investment Intelligence
==============================================================================
Regresión logística BAYESIANA, siguiendo las prácticas de modelado descriptas
en "Pattern Recognition and Machine Learning" de Christopher M. Bishop,
Capítulo 4 (Linear Models for Classification), sección 4.5 (Bayesian Logistic
Regression) en particular.

Qué distingue este enfoque de un "sklearn.LogisticRegression().fit()" genérico:

  1. MAP en vez de máxima verosimilitud pura — la regularización L2 se
     interpreta como un prior Gaussiano isotrópico sobre los pesos,
     N(w | 0, alpha^-1 I) (Bishop, ec. 4.142 y contexto de 4.5).

  2. El ajuste se hace con IRLS (Iterative Reweighted Least Squares),
     implementado acá desde cero en numpy siguiendo la derivación exacta
     de Bishop (ecs. 4.99-4.100, extendida con el término de prior) — no
     es una caja negra de una librería.

  3. La fuerza de regularización (alpha) NO se elige a mano: se busca por
     validación cruzada respetando el orden temporal (walk-forward, nunca
     folds aleatorios — mezclar el tiempo en una serie financiera es
     hacer trampa, el modelo vería el "futuro" durante el entrenamiento).

  4. Aproximación de Laplace (Bishop, sección 4.5.1) alrededor del máximo
     a posteriori, para obtener la covarianza posterior S_N sobre los
     pesos — es decir, no solo "cuáles son los pesos", sino "cuán seguro
     está el modelo de esos pesos".

  5. Distribución predictiva Bayesiana (Bishop, sección 4.5.2, ec. 4.153):
     la probabilidad final para un caso nuevo NO es simplemente
     sigmoid(w · x) — incorpora la incertidumbre de los pesos, así que
     cuando el modelo está poco seguro, la predicción se "encoge" sola
     hacia 50%, en vez de sobre-confiar con una probabilidad extrema.

  6. La métrica principal para elegir el modelo es log-loss (entropía
     cruzada), no accuracy — Bishop argumenta desde teoría de decisión
     (capítulo 1.5) que para modelos probabilísticos hay que evaluar
     la calidad de la probabilidad en sí, no solo si el 0/1 final acertó.

Por qué corre acá y no en el chat de Claude:
Este script necesita conexión a internet real para bajar el historial de
CoinGecko. El entorno donde chateás con Claude tiene acceso a internet
restringido y no puede alcanzar esa API — por eso este paso se hace en tu
compu, con el mismo Python que ya usaste para el EDA original.

Requisitos (una sola vez):
    pip install pandas numpy scikit-learn requests

Uso:
    python entrenar_modelo_predictivo.py
"""

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss

np.set_printoptions(precision=4, suppress=True)

# ============================================================
# 1. DESCARGAR HISTORIAL REAL DE ETH
# ============================================================
print("Descargando historial de precios de ETH (2 años, diario) desde CoinGecko...")
url = "https://api.coingecko.com/api/v3/coins/ethereum/market_chart"
params = {"vs_currency": "usd", "days": 730, "interval": "daily"}
resp = requests.get(url, params=params, timeout=30)
resp.raise_for_status()
data = resp.json()

precios_serie = [p[1] for p in data["prices"]]
fechas = [pd.to_datetime(p[0], unit="ms") for p in data["prices"]]
df = pd.DataFrame({"fecha": fechas, "precio": precios_serie}).set_index("fecha")
print(f"Descargados {len(df)} días de precio ({df.index.min().date()} a {df.index.max().date()}).")


# ============================================================
# 2. LOS 4 INDICADORES TÉCNICOS (mismas fórmulas que en n8n)
# ============================================================
def calcular_features(precios_arr, i):
    """
    Calcula los 4 scores técnicos para el día i, usando SOLO datos hasta
    ese día — nunca mira el futuro (look-ahead bias es el error más común
    y más grave al construir modelos predictivos con series de tiempo).
    Devuelve valores en escala 0-1 (ya normalizados), listos para el modelo.
    """
    ventana = precios_arr[: i + 1]
    if len(ventana) < 91:
        return None

    retornos = np.diff(ventana) / ventana[:-1]

    # 1. Volatilidad: 30d reciente vs. 90d base
    vol_reciente = np.std(retornos[-30:])
    vol_base = np.std(retornos[-90:])
    ratio_vol = vol_reciente / vol_base if vol_base > 0 else 1
    score_vol = np.clip(100 - max(0, ratio_vol - 1) * 150, 0, 100)

    # 2. Tendencia: precio vs SMA7 vs SMA30
    precio_actual = ventana[-1]
    sma7 = np.mean(ventana[-7:])
    sma30 = np.mean(ventana[-30:])
    señales = int(precio_actual > sma7) + int(sma7 > sma30) + int(precio_actual > sma30)
    score_tendencia = (señales / 3) * 100
    tendencia_alcista = precio_actual > sma30

    # 3. Momentum / autocorrelación (lag-1), ajustado por dirección
    retornos90 = retornos[-90:]
    mean_r = np.mean(retornos90)
    num = np.sum((retornos90[1:] - mean_r) * (retornos90[:-1] - mean_r))
    den = np.sum((retornos90 - mean_r) ** 2)
    autocorr = num / den if den > 0 else 0
    if autocorr >= 0:
        score_momentum = 50 + autocorr * 200 if tendencia_alcista else 50 - autocorr * 200
    else:
        score_momentum = 50 - autocorr * 100
    score_momentum = np.clip(score_momentum, 0, 100)

    # 4. Distancia al ATH — usando el máximo HASTA ese día, no el ATH real/futuro
    ath_hasta_hoy = np.max(ventana)
    caida_ath = 1 - (precio_actual / ath_hasta_hoy)
    score_ath = np.clip(caida_ath * 100, 0, 100)

    return np.array([score_vol, score_tendencia, score_momentum, score_ath]) / 100.0


# ============================================================
# 3. ARMAR EL DATASET (features de hoy + resultado a 5 días)
# ============================================================
HORIZONTE = 5
filas_X, filas_y = [], []
precios_arr = df["precio"].values
for i in range(91, len(precios_arr) - HORIZONTE):
    feats = calcular_features(precios_arr, i)
    if feats is None:
        continue
    subio = 1 if precios_arr[i + HORIZONTE] > precios_arr[i] else 0
    filas_X.append(feats)
    filas_y.append(subio)

X_all = np.array(filas_X)
y_all = np.array(filas_y)
print(f"\nDataset armado: {len(X_all)} ejemplos, target = ¿subió a los {HORIZONTE} días?")
print(f"Distribución del target: {y_all.mean():.1%} positivos / {1-y_all.mean():.1%} negativos")


# ============================================================
# 4. REGRESIÓN LOGÍSTICA BAYESIANA — IRLS + Laplace (Bishop 4.3.3 / 4.5)
# ============================================================
def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-np.clip(a, -500, 500)))


def fit_map_irls(Phi, t, alpha, max_iter=100, tol=1e-6):
    """
    Ajusta w_MAP por IRLS, siguiendo Bishop ecs. 4.99-4.100, con un término
    de regularización L2 (equivalente a un prior N(0, alpha^-1 I) sobre w).
    Phi: matriz de diseño (N x M), YA con columna de bias (unos) incluida.
    Devuelve: w_map, S_N (covarianza posterior via Laplace, ec. 4.143).
    """
    N, M = Phi.shape
    w = np.zeros(M)
    for _ in range(max_iter):
        a = Phi @ w
        y = sigmoid(a)
        R = np.diag(y * (1 - y) + 1e-9)  # + eps: estabilidad numérica

        grad = Phi.T @ (y - t) + alpha * w                    # Bishop ec. 4.96 + prior
        H = Phi.T @ R @ Phi + alpha * np.eye(M)                # Bishop ec. 4.97 + prior

        w_new = w - np.linalg.solve(H, grad)                   # paso de Newton-Raphson (IRLS)
        if np.linalg.norm(w_new - w) < tol:
            w = w_new
            break
        w = w_new

    # Aproximación de Laplace: covarianza posterior = inversa del Hessiano en el MAP
    a = Phi @ w
    y = sigmoid(a)
    R = np.diag(y * (1 - y) + 1e-9)
    H = Phi.T @ R @ Phi + alpha * np.eye(M)
    S_N = np.linalg.inv(H)                                     # Bishop ec. 4.143
    return w, S_N


def predictive_prob_bayes(phi, w, S_N):
    """
    Distribución predictiva incorporando la incertidumbre de los pesos
    (Bishop ec. 4.151-4.153, aproximación probit de la convolución
    sigmoid-Gaussiana). Cuando sigma_a^2 es grande (modelo inseguro),
    kappa < 1 y la probabilidad se encoge hacia 0.5.
    """
    mu_a = phi @ w
    sigma_a2 = phi @ S_N @ phi
    kappa = 1.0 / np.sqrt(1.0 + np.pi * sigma_a2 / 8.0)
    return sigmoid(kappa * mu_a), sigma_a2


def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


# ============================================================
# 5. SPLIT: hold-out final (nunca se toca hasta el final) + CV temporal
#    dentro del set de entrenamiento, para elegir alpha (Bishop cap. 1.3)
# ============================================================
n = len(X_all)
split_final = int(n * 0.85)
X_trainval, y_trainval = X_all[:split_final], y_all[:split_final]
X_test, y_test = X_all[split_final:], y_all[split_final:]

alphas_candidatos = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]
tscv = TimeSeriesSplit(n_splits=5)

print("\n===== Selección de alpha (regularización) por validación cruzada temporal =====")
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
    print(f"  alpha={alpha:>6.1f}  →  log-loss promedio (5 folds temporales) = {avg_loss:.4f}")
    if avg_loss < mejor_logloss:
        mejor_logloss, mejor_alpha = avg_loss, alpha

print(f"\nAlpha elegido: {mejor_alpha} (menor log-loss en validación cruzada)")


# ============================================================
# 6. MODELO FINAL — entrenado en todo el set de train+val, evaluado
#    UNA SOLA VEZ en el hold-out final (nunca visto hasta acá)
# ============================================================
Phi_trainval = add_bias(X_trainval)
Phi_test = add_bias(X_test)
w_final, S_N_final = fit_map_irls(Phi_trainval, y_trainval, mejor_alpha)

probs_bayes = np.array([predictive_prob_bayes(phi, w_final, S_N_final)[0] for phi in Phi_test])
probs_map_puro = sigmoid(Phi_test @ w_final)  # sin incorporar incertidumbre, para comparar

probs_bayes_clip = np.clip(probs_bayes, 1e-6, 1 - 1e-6)
probs_map_clip = np.clip(probs_map_puro, 1e-6, 1 - 1e-6)

pred_bayes = (probs_bayes >= 0.5).astype(int)
baseline = max(y_test.mean(), 1 - y_test.mean())

print("\n===== RESULTADOS SOBRE EL HOLD-OUT FINAL (nunca visto durante el entrenamiento) =====")
print(f"Accuracy (predictiva bayesiana):        {accuracy_score(y_test, pred_bayes):.1%}")
print(f"Accuracy de un baseline ingenuo:        {baseline:.1%}  (predecir siempre la clase más común)")
print(f"Log-loss (predictiva bayesiana):        {log_loss(y_test, probs_bayes_clip, labels=[0,1]):.4f}")
print(f"Log-loss (MAP puro, sin incertidumbre): {log_loss(y_test, probs_map_clip, labels=[0,1]):.4f}")
print(f"Brier score (calibración, bayesiana):   {brier_score_loss(y_test, probs_bayes):.4f}")

if log_loss(y_test, probs_bayes_clip, labels=[0, 1]) < log_loss(y_test, probs_map_clip, labels=[0, 1]):
    print(
        "\n✓ Incorporar la incertidumbre de los pesos (enfoque bayesiano) mejoró la\n"
        "  calidad probabilística respecto a usar el MAP puro — el efecto que predice\n"
        "  Bishop en la sección 4.5.2: cuando el modelo no está seguro, encoger la\n"
        "  probabilidad hacia 0.5 evita sobre-confiar y mejora el log-loss."
    )

if accuracy_score(y_test, pred_bayes) <= baseline + 0.02:
    print(
        "\n⚠️  El modelo NO supera de forma clara a un baseline ingenuo en accuracy.\n"
        "    Es esperable en mercados razonablemente eficientes con solo 4 indicadores\n"
        "    técnicos — no es un error del código, es una medición honesta de cuánta\n"
        "    ventaja predictiva hay realmente. Igual podés usarlo como una señal más,\n"
        "    pero no le des más peso del que estos números muestran."
    )

# ============================================================
# 7. EXPORTAR — pesos MAP + covarianza posterior, listos para n8n
# ============================================================
print("\n===== PARA PEGAR EN EL NODO CODE DE N8N =====")
print(f"""
// Pesos entrenados (MAP) — orden: [bias, volatilidad, tendencia, momentum, ath]
const pesosModelo = {list(np.round(w_final, 5))};

// Covarianza posterior S_N (5x5, aproximación de Laplace) — fila por fila
const covarianzaModelo = [
{chr(10).join('  ' + str(list(np.round(row, 6))) + ',' for row in S_N_final)}
];

// Precisión medida en el hold-out final, para mostrar en el mail:
const accuracyModelo = {accuracy_score(y_test, pred_bayes):.4f};
const logLossModelo = {log_loss(y_test, probs_bayes_clip, labels=[0,1]):.4f};
""")
