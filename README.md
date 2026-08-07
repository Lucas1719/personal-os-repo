# Personal OS — Sistema de Automatización con IA

Sistema personal de automatización construido con **Claude (Desktop + API + Code)**, **n8n** y un stack de APIs gratuitas, diseñado para automatizar gestión de correo, seguimiento de inversión en cripto, estudio y búsqueda de trabajo freelance.

Este repo documenta el módulo más maduro del sistema — **Investment Intelligence** — de punta a punta: desde el pipeline de datos hasta un modelo predictivo bayesiano entrenado, validado y monitoreado en producción.

> 🧠 Portfolio técnico de **[tu nombre]** — AI Operator / Prompt Engineer freelance. Este mismo proyecto es un caso de referencia del tipo de trabajo que ofrezco: pipelines de datos automatizados, integraciones de IA, y sistemas de reporting inteligente con evaluación estadística honesta.

---

## Qué hace el sistema

Todos los días a las 8 AM, un flujo de **n8n** se dispara solo y:

1. Trae precio, volumen e historial diario de ETH y BTC en vivo desde **Binance**
2. Calcula 12 indicadores técnicos (volatilidad, tendencia, momentum, RSI, MACD, Bandas de Bollinger, correlación con BTC, estacionalidad, entre otros)
3. Corre esos indicadores a través de un **modelo predictivo entrenado** (regresión logística bayesiana) para estimar la probabilidad de que el precio suba a 5 días, con un semáforo de 3 colores (🟢🟡🔴)
4. **Verifica automáticamente** la predicción de hace 5 días contra el precio real, y acumula una **tasa de aciertos bayesiana en producción** (con su intervalo de credibilidad, no un número pelado)
5. Consulta el índice de sentimiento del mercado (Fear & Greed)
6. Le pasa todo eso a **Claude**, que redacta un reporte diario en lenguaje natural con una recomendación operativa (comprar / vender / no hacer nada), objetivos de entrada/salida calculados, y los montos exactos listos para cargar en Binance
7. Envía el reporte por mail y guarda un log histórico en Google Sheets

El operador decide manualmente qué hacer con esa información — el sistema **propone, nunca ejecuta solo**. Las operaciones se registran en un panel HTML propio (`web-apps/eth-trading-log-bayes.html`) que acumula la tasa de aciertos real del operador con el tiempo.

---

## Stack técnico

| Capa | Herramientas |
|---|---|
| Orquestación | n8n (self-hosted) |
| LLM | Claude API (Anthropic) |
| Datos de mercado | Binance API (Klines + ticker 24hr) — misma fuente para entrenamiento y producción |
| Sentimiento | Alternative.me Fear & Greed Index |
| Persistencia | Google Sheets (log diario + pesos del modelo + historial de verificaciones) |
| Modelo predictivo | Python — numpy (implementación propia), pandas, scikit-learn (solo utilidades de CV) |
| Apps de apoyo | HTML/CSS/JS vanilla (sin frameworks) |

---

## El modelo predictivo — Regresión Logística Bayesiana

En vez de usar una librería de ML como caja negra, el modelo está implementado **desde cero en numpy**, siguiendo las prácticas descriptas en *Pattern Recognition and Machine Learning* (Christopher M. Bishop), capítulo 4.5:

- **IRLS** (Iterative Reweighted Least Squares) para el ajuste MAP con regularización L2 (prior Gaussiano)
- **Aproximación de Laplace** para obtener la incertidumbre sobre los pesos aprendidos, no solo su valor puntual
- **Distribución predictiva Bayesiana** — la probabilidad final se "encoge" hacia 50% cuando el modelo está inseguro, en vez de sobre-confiar
- **Validación cruzada respetando el orden temporal** (walk-forward, nunca folds aleatorios) para elegir la regularización

### Lo que se probó y se descartó, con evidencia

Este proyecto documenta tanto lo que funcionó como lo que **no** funcionó — y por qué:

- **Automatic Relevance Determination (ARD)**, evaluada como candidata para selección automática de features (`modelo-predictivo/entrenar_modelo_predictivo.py`). Se descartó: peor log-loss que la regularización fija, y la selección de features resultó inestable entre folds de validación cruzada.
- **Comparación de horizontes de predicción** (3, 5 y 7 días — `modelo-predictivo/comparar_horizontes.py`): los tres quedaron pegados a `ln(2) ≈ 0.693`, el log-loss de un modelo que no sabe nada. No hay evidencia de que ningún horizonte sea mejor que otro con los datos actuales — se mantuvo 5 días por no haber motivo real para cambiarlo.
- **Accuracy en hold-out temporal**: 48.7% vs. 54.7% de un baseline ingenuo (predecir siempre la clase mayoritaria). El modelo **no supera** a no hacer nada inteligente.
- **Test de verificación con observaciones independientes** (`modelo-predictivo/simular_produccion_historica.py`, espaciando las muestras cada 5 días para evitar ventanas superpuestas): 62.2% de aciertos sobre 45 observaciones, con un intervalo de credibilidad del 95% de [47.5%, 74.9%] — prometedor, pero el intervalo todavía toca 50%, así que no es concluyente. Se sigue trackeando en vivo para ver si el intervalo se angosta con más datos.

Este último punto usa la misma actualización Bayesiana (Beta-Binomial) para la tasa de aciertos que el modelo usa para sus pesos — no es un cociente simple, es una estimación con su propia incertidumbre.

---

## Estructura del repo

```
docs/               → documentación del proyecto (visión, estrategia, case study del EDA)
web-apps/            → herramientas HTML standalone (panel de trading, resumidor de estudio, guías)
modelo-predictivo/   → scripts de entrenamiento, comparación de horizontes, simulación retroactiva
n8n/                 → código del nodo de procesamiento (JS) que corre en producción
```

---

## Otros módulos del sistema

Este repo se enfoca en Investment Intelligence, pero el Personal OS completo incluye:

- **📧 Mail Intelligence** — triage automático de correos, agenda, resúmenes de reuniones
- **📚 Learning Intelligence** — resumidor de estudio + flashcards interactivas ([`web-apps/resumidor-estudio.html`](web-apps/resumidor-estudio.html))
- **💼 Career Intelligence** — búsqueda de trabajo automatizada y case studies de portfolio (este repo es uno de ellos)

---

## Nota

Este proyecto usa capital y operaciones **ficticias** (paper trading / Binance Demo Trading) mientras se valida el sistema. Ningún análisis acá constituye asesoramiento financiero.

---

*Construido con Claude + n8n · Agosto 2026*
