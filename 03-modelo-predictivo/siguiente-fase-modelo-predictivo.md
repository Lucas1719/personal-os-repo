# Siguiente fase — Modelo predictivo ETH (Investment Intelligence)

*Personal OS · Brief de traspaso para Claude Code · Agosto 2026*

## Contexto — qué existe hoy

Hay un modelo predictivo **vigente** en producción (`entrenar_modelo_predictivo.py`, en la carpeta del proyecto):

- Regresión logística **bayesiana** (IRLS + aproximación de Laplace, siguiendo Bishop, *Pattern Recognition and Machine Learning*, cap. 4.5)
- **4 features**, elegidas a mano por intuición de dominio, no por selección sistemática:
  1. Volatilidad reciente (30d) vs. base (90d)
  2. Régimen de tendencia (precio vs. SMA7 vs. SMA30)
  3. Momentum / autocorrelación (lag-1), ajustado por dirección de tendencia
  4. Distancia ajustada al ATH
- Entrenado con **~270 ejemplos (1 año de historial diario de ETH vía CoinGecko)**
- Resultado medido en hold-out temporal: **accuracy 39.0% vs. baseline ingenuo 61.0%** — el modelo no muestra ventaja predictiva demostrada todavía. Esto está documentado y comunicado con honestidad en el reporte diario (sección "🧪 Modelo predictivo — EXPERIMENTAL").
- Los pesos entrenados viven en una pestaña de Google Sheets ("Modelo Predictivo"), que n8n lee todos los días — no están hardcodeados en el flujo.

## Objetivo de esta fase

Mejorar el modelo en dos frentes, **en este orden**:

### 1. Más historial real (antes que más features)

- **Problema actual:** CoinGecko gratuito limita a ~365 días diarios — la muestra es demasiado chica para conclusiones confiables (un hold-out de ~40 ejemplos tiene margen de error enorme).
- **Solución propuesta:** cambiar la fuente de datos del script de entrenamiento de CoinGecko a **Binance Klines** (`GET https://api.binance.com/api/v3/klines?symbol=ETHUSDT&interval=1d&startTime=...&endTime=...`), que tiene historial diario de ETH/USDT desde 2017, sin necesitar API key.
- **Matiz importante a mantener:** más datos no es automáticamente mejor — el ETH 2017-2020 (época de ICOs) puede no ser representativo del régimen de mercado actual. Evaluar con el operador un rango razonable (sugerido: 3-4 años) en vez de traer el máximo histórico posible sin criterio.
- Klines de Binance vienen en un formato distinto a CoinGecko (arrays de 12 campos: open time, open, high, low, close, volume, close time, ...) — hay que adaptar el parseo, no es un simple cambio de URL.

### 2. Más features candidatas (feature engineering)

Ampliar el menú de indicadores posibles más allá de los 4 actuales. Candidatas sugeridas por el operador y ya alineadas con hallazgos del EDA original (`case-study-btc-eth-eda.docx`):

- RSI, MACD, Bandas de Bollinger (momentum/volatilidad más sofisticados que los actuales)
- Features de volumen — el EDA original ya documentó que "el volumen confirma o desmiente los movimientos de precio"; hoy esa señal no se usa en el modelo
- Correlación / fuerza relativa contra BTC — el EDA encontró que la correlación BTC-ETH se estrecha en crisis; podría ser señal de régimen
- Estacionalidad simple (día de la semana, fin de mes)
- Opcional / si hay fuente disponible: datos on-chain de ETH (gas fees, direcciones activas)

### 3. Selección automática de features — Automatic Relevance Determination (ARD)

En vez de que el operador o Claude elijan a mano cuáles de las features nuevas usar, implementar **ARD** (Bishop, PRML, contexto de sección 3.5 / framework de evidencia, y cap. 7 — mismo principio detrás de las Relevance Vector Machines):

- En vez de un único `alpha` global de regularización (como tiene el modelo actual), cada feature `i` tiene su propio `alpha_i`
- Procedimiento iterativo, alternando:
  1. Con los `alpha_i` actuales fijos, ajustar `w` por IRLS (igual que ya está implementado)
  2. Con `w` y `S_N` (covarianza posterior de Laplace) fijos, actualizar cada `alpha_i` por maximización de evidencia (tipo-II ML): `alpha_i_nuevo = gamma_i / w_i^2`, donde `gamma_i = 1 - alpha_i * S_N[i,i]`
  3. Repetir hasta convergencia
- Las features irrelevantes terminan con `alpha_i → muy grande`, lo que empuja su `w_i → 0` — quedan "apagadas" solas, sin que nadie decida a mano qué sacar
- **Mantener el mismo estilo de implementación** que ya tiene `entrenar_modelo_predictivo.py`: numpy desde cero (no sklearn como caja negra para el modelo en sí), con comentarios que referencien las ecuaciones de Bishop correspondientes
- Mantener también la validación cruzada temporal (walk-forward, nunca folds aleatorios) y el reporte de log-loss como métrica principal, igual que en la versión actual

## Qué NO cambiar

- El resto del sistema (n8n, el panel bayesiano, la estructura del mail) no necesita tocarse en esta fase — solo el script de entrenamiento y, al final, actualizar la fila de Google Sheets con los nuevos pesos (que ahora van a ser más de 5, según cuántas features sobrevivan el ARD)
- Seguir exportando al final los pesos + covarianza + accuracy/logloss en el mismo formato JSON listo para pegar en la pestaña "Modelo Predictivo" de Sheets
- Mantener la honestidad del reporte: si el modelo sigue sin superar al baseline incluso con más datos y más features, comunicarlo igual de claro que hoy — no maquillar el resultado

## Cómo continuar

1. Confirmar con el operador el rango de años de historial a usar (sugerido 3-4 años vía Binance Klines)
2. Adaptar la descarga y el parseo de datos históricos
3. Implementar las features candidatas nuevas (una por vez, verificando cada fórmula antes de sumar la siguiente)
4. Implementar ARD sobre el conjunto ampliado de features
5. Entrenar, evaluar en hold-out temporal, y comparar honestamente contra el modelo actual (accuracy 39%, log-loss 0.7286) — ¿mejoró de verdad, o es ruido de muestra chica otra vez?
6. Exportar los pesos nuevos en formato JSON listo para Sheets
7. Volver al chat de Claude (no Claude Code) para actualizar la fila de Sheets y confirmar que n8n sigue leyendo bien el nuevo formato (si cambió la cantidad de features, hay que ajustar también el `phiHoy` del nodo Code en n8n)
