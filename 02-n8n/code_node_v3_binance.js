const allItems = $input.all();

const tickerItem = allItems.find(i => i.json?.lastPrice);
const fgItem = allItems.find(i => i.json?.data?.[0]?.value_classification);
const modeloItem = allItems.find(i => i.json?.pesos_json);

if (!tickerItem || !modeloItem) {
  throw new Error(`Faltan datos base. Items: ${allItems.length}. ` +
    `Ticker: ${!!tickerItem}, F&G: ${!!fgItem}, Modelo: ${!!modeloItem}`);
}

const ticker = tickerItem.json;
const fg = fgItem ? fgItem.json.data[0] : { value: null, value_classification: null };

// Klines — referenciados por NOMBRE (vienen como 1000 items c/u, no se pueden
// mezclar con allItems.find porque ETH y BTC tienen exactamente la misma forma)
let klinesETHItems, klinesBTCItems;
try {
  klinesETHItems = $('Klines ETH').all();
  klinesBTCItems = $('Klines BTC').all();
} catch (e) {
  throw new Error('No se pudo leer "Klines ETH" o "Klines BTC" — confirmá que esos ' +
    'nodos existen con ese nombre exacto y están conectados al Merge.');
}

// Cada kline: [0]openTime [1]open [2]high [3]low [4]close [5]volumen [6]closeTime ...
function parsearKlines(items){
  const filas = items.map(i => i.json);
  filas.sort((a,b) => Number(a[0]) - Number(b[0])); // por las dudas, orden cronológico
  return {
    precios: filas.map(k => parseFloat(k[4])),
    volumenes: filas.map(k => parseFloat(k[5])),
  };
}

const { precios: preciosHistoricos, volumenes: volumenHistorico } = parsearKlines(klinesETHItems);
const { precios: preciosBTC } = parsearKlines(klinesBTCItems);

// ============================================================
// DATOS DEL TICKER — reemplazan lo que antes daba CoinGecko
// ============================================================
const precioActualUSD = parseFloat(ticker.lastPrice);
const cambio24h = parseFloat(ticker.priceChangePercent);
const volumen24hUSD = parseFloat(ticker.quoteVolume);
const high24h = parseFloat(ticker.highPrice);
const low24h = parseFloat(ticker.lowPrice);

// ATH — Binance con 1000 días no alcanza al ATH real (nov 2021). Se deja
// como constante documentada, solo para mostrar en el mail (NO se usa
// para el feature del modelo, que sigue usando el máximo de la ventana
// disponible, igual que hizo el script de entrenamiento).
const ATH_HISTORICO = 4946;
const athPct = ((precioActualUSD - ATH_HISTORICO) / ATH_HISTORICO) * 100;

// cambio 7d / 30d — CoinGecko los daba directo, ahora se calculan de los Klines
const idxHoy = preciosHistoricos.length - 1;
const precioHace7d = preciosHistoricos[idxHoy - 7];
const precioHace30d = preciosHistoricos[idxHoy - 30];
const cambio7d = ((precioActualUSD - precioHace7d) / precioHace7d) * 100;
const cambio30d = ((precioActualUSD - precioHace30d) / precioHace30d) * 100;

// market cap — Binance no lo provee (requiere supply circulante). Aproximado,
// no exacto — si hace falta precisión real, hay que sumar otra fuente.
const SUPPLY_APROX_ETH = 120500000;
const marketCapAprox = precioActualUSD * SUPPLY_APROX_ETH;

// ============================================================
// FUNCIONES BASE
// ============================================================
function mean(arr){ return arr.reduce((a,b)=>a+b,0) / arr.length; }
function stdDevPop(arr){
  const m = mean(arr);
  return Math.sqrt(arr.reduce((a,b)=>a + Math.pow(b-m,2), 0) / arr.length);
}
function emaAdjustFalse(arr, span){
  const alpha = 2 / (span + 1);
  let ema = arr[0];
  const out = [ema];
  for (let i = 1; i < arr.length; i++){
    ema = alpha * arr[i] + (1 - alpha) * ema;
    out.push(ema);
  }
  return out;
}
function pearsonCorr(x, y){
  const mx = mean(x), my = mean(y);
  let num = 0, denx = 0, deny = 0;
  for (let i = 0; i < x.length; i++){
    num += (x[i]-mx) * (y[i]-my);
    denx += Math.pow(x[i]-mx, 2);
    deny += Math.pow(y[i]-my, 2);
  }
  const den = Math.sqrt(denx * deny);
  return den > 0 ? num/den : NaN;
}

// ============================================================
// LAS 12 FEATURES — orden exacto: ["volatilidad","tendencia","momentum","ath",
// "rsi","macd","bollinger","volumen","corr_btc","estacionalidad_sin",
// "estacionalidad_cos","fin_de_mes"]
// ============================================================
const ventana = preciosHistoricos;
const volVentana = volumenHistorico;
const ventanaBTC = preciosBTC;

const retornos = [];
for (let i = 1; i < ventana.length; i++) retornos.push((ventana[i]-ventana[i-1]) / ventana[i-1]);
const precioActual = ventana[ventana.length - 1];

// 1. Volatilidad
const volReciente = stdDevPop(retornos.slice(-30));
const volBase = stdDevPop(retornos);
const ratioVol = volBase > 0 ? volReciente / volBase : 1;
const scoreVol = Math.max(0, Math.min(100, 100 - Math.max(0,(ratioVol - 1)) * 150));

// 2. Tendencia
const sma7 = mean(ventana.slice(-7));
const sma30 = mean(ventana.slice(-30));
let señales = 0;
if (precioActual > sma7) señales++;
if (sma7 > sma30) señales++;
if (precioActual > sma30) señales++;
const scoreTendencia = (señales / 3) * 100;
const tendenciaAlcista = precioActual > sma30;
const regimen = señales === 3 ? 'alcista' : señales === 0 ? 'bajista' : 'lateral';

// 3. Momentum / autocorrelación
const retornos90 = retornos.slice(-90);
const meanR = mean(retornos90);
let numMom = 0, denMom = 0;
for (let i = 1; i < retornos90.length; i++) numMom += (retornos90[i]-meanR) * (retornos90[i-1]-meanR);
for (let i = 0; i < retornos90.length; i++) denMom += Math.pow(retornos90[i]-meanR, 2);
const autocorr = denMom > 0 ? numMom/denMom : 0;
let scoreMomentum;
if (autocorr >= 0) scoreMomentum = tendenciaAlcista ? 50 + autocorr*200 : 50 - autocorr*200;
else scoreMomentum = 50 - autocorr*100;
scoreMomentum = Math.max(0, Math.min(100, scoreMomentum));

// 4. Distancia al ATH (de la VENTANA disponible — consistente con el entrenamiento)
const athHastaHoy = Math.max(...ventana);
const caidaATH = 1 - (precioActual / athHastaHoy);
const scoreATH = Math.max(0, Math.min(100, caidaATH * 100));

// 5. RSI(14)
const r14 = retornos.slice(-14);
const avgGain = mean(r14.map(r => Math.max(r, 0)));
const avgLoss = mean(r14.map(r => Math.max(-r, 0)));
const rs = avgLoss > 0 ? avgGain/avgLoss : Infinity;
const scoreRSI = isFinite(rs) ? 100 - 100/(1+rs) : 100;

// 6. MACD
const ema12 = emaAdjustFalse(ventana, 12);
const ema26 = emaAdjustFalse(ventana, 26);
const macdLine = ema12.map((v,i) => v - ema26[i]);
const signalLine = emaAdjustFalse(macdLine, 9);
const histograma = macdLine[macdLine.length-1] - signalLine[signalLine.length-1];
const macdPct = histograma / precioActual;
const scoreMACD = (Math.tanh(macdPct * 50) + 1) / 2 * 100;

// 7. Bollinger %B
const sma20 = mean(ventana.slice(-20));
const std20 = stdDevPop(ventana.slice(-20));
const bandaSup = sma20 + 2*std20;
const bandaInf = sma20 - 2*std20;
const pctB = bandaSup > bandaInf ? (precioActual - bandaInf) / (bandaSup - bandaInf) : 0.5;
const scoreBollinger = Math.max(0, Math.min(1, pctB)) * 100;

// 8. Volumen relativo
const vol7 = mean(volVentana.slice(-7));
const vol30 = mean(volVentana.slice(-30));
const ratioVolumen = vol30 > 0 ? vol7/vol30 : 1;
const scoreVolumen = (Math.tanh((ratioVolumen - 1) * 2) + 1) / 2 * 100;

// 9. Correlación ETH-BTC (30d)
const retornosBTC = [];
for (let i = 1; i < ventanaBTC.length; i++) retornosBTC.push((ventanaBTC[i]-ventanaBTC[i-1]) / ventanaBTC[i-1]);
const nCorr = Math.min(30, retornos.length, retornosBTC.length);
let corr = 0;
if (nCorr >= 5){
  const c = pearsonCorr(retornos.slice(-nCorr), retornosBTC.slice(-nCorr));
  corr = isNaN(c) ? 0 : c;
}
const scoreCorrBTC = (corr + 1) / 2 * 100;

// 10-11. Estacionalidad
const hoyDate = new Date();
const jsDay = hoyDate.getDay();
const dow = (jsDay + 6) % 7;
const scoreDowSin = (Math.sin(2*Math.PI*dow/7) + 1) / 2 * 100;
const scoreDowCos = (Math.cos(2*Math.PI*dow/7) + 1) / 2 * 100;

// 12. Fin de mes
const ultimoDiaMes = new Date(hoyDate.getFullYear(), hoyDate.getMonth()+1, 0).getDate();
const scoreFinMes = (ultimoDiaMes - hoyDate.getDate()) < 3 ? 100 : 0;

const scoreTecnico = Math.round(scoreVol*0.30 + scoreTendencia*0.30 + scoreMomentum*0.20 + scoreATH*0.20);

// ============================================================
// MODELO PREDICTIVO — 13 pesos (bias + 12 features)
// ============================================================
const pesosModelo = JSON.parse(modeloItem.json.pesos_json);
const covarianzaModelo = JSON.parse(modeloItem.json.covarianza_json);
const accuracyModelo = parseFloat(modeloItem.json.accuracy);
const logLossModelo = parseFloat(modeloItem.json.logloss);
const baselineModelo = parseFloat(modeloItem.json.baseline);
const descripcionModelo = modeloItem.json.descripcion_modelo || 'historial reciente de ETH y BTC (Binance)';

function sigmoidJS(a){ return 1 / (1 + Math.exp(-a)); }
function prediccionBayesiana(phi, w, S_N){
  let mu_a = 0;
  for (let i = 0; i < phi.length; i++) mu_a += phi[i] * w[i];
  let sigma_a2 = 0;
  for (let i = 0; i < phi.length; i++){
    let fila = 0;
    for (let j = 0; j < phi.length; j++) fila += S_N[i][j] * phi[j];
    sigma_a2 += phi[i] * fila;
  }
  const kappa = 1 / Math.sqrt(1 + Math.PI * sigma_a2 / 8);
  return sigmoidJS(kappa * mu_a);
}

const phiHoy = [
  1,
  scoreVol/100, scoreTendencia/100, scoreMomentum/100, scoreATH/100,
  scoreRSI/100, scoreMACD/100, scoreBollinger/100, scoreVolumen/100, scoreCorrBTC/100,
  scoreDowSin/100, scoreDowCos/100, scoreFinMes/100
];
const probabilidadSube = Math.round(prediccionBayesiana(phiHoy, pesosModelo, covarianzaModelo) * 100);

// Semáforo — umbrales angostos porque los pesos del modelo son chicos y
// las predicciones rara vez se alejan mucho de 50%
const colorModelo = probabilidadSube > 53 ? 'verde' : probabilidadSube < 47 ? 'rojo' : 'amarillo';
const colorModeloEmoji = colorModelo === 'verde' ? '🟢' : colorModelo === 'rojo' ? '🔴' : '🟡';

// Verificación retrospectiva — ¿la predicción de hace 5 días acertó?
const ZONA_MUERTA_PCT = 3; // movimientos menores a esto (en cualquier dirección) no cuentan ni como acierto ni como error
let verifDisponible = false, verifFecha = null, verifProbabilidad = null, verifColor = null,
    verifPrecioEntonces = null, verifCambioPct = null, verifResultado = 'sin_datos';
try {
  const logHace5Dias = $('Leer Log Hace 5 Días').first();
  if (logHace5Dias && logHace5Dias.json && logHace5Dias.json.probabilidadSube) {
    const probAnterior = parseFloat(logHace5Dias.json.probabilidadSube);
    const precioAnterior = parseFloat(logHace5Dias.json.precio_usd);
    const colorAnterior = probAnterior > 53 ? 'verde' : probAnterior < 47 ? 'rojo' : 'amarillo';
    const cambioPct = ((precioActualUSD - precioAnterior) / precioAnterior) * 100;
    const subioReal = precioActualUSD > precioAnterior;

    verifDisponible = true;
    verifFecha = logHace5Dias.json.fecha;
    verifProbabilidad = probAnterior;
    verifColor = colorAnterior;
    verifPrecioEntonces = precioAnterior;
    verifCambioPct = +cambioPct.toFixed(2);

    if (colorAnterior === 'amarillo') {
      // Amarillo = "no espero un movimiento claro". Se evalúa distinto:
      // si el precio efectivamente se quedó quieto, acertó la incertidumbre.
      // Si pegó un salto grande para cualquier lado, no lo vio venir.
      verifResultado = Math.abs(cambioPct) < ZONA_MUERTA_PCT ? 'acierto' : 'error';
    } else if (Math.abs(cambioPct) < ZONA_MUERTA_PCT) {
      verifResultado = 'movimiento_insignificante'; // verde/rojo con movimiento chico: no hay forma de evaluar
    } else if (colorAnterior === 'verde') {
      verifResultado = subioReal ? 'acierto' : 'error';
    } else {
      verifResultado = subioReal ? 'error' : 'acierto';
    }
  }
} catch (e) {
  verifDisponible = false; // todavía no hay 5 días de historial acumulado, es normal al principio
}

// Tasa de aciertos acumulada en producción — leyendo TODO el historial de
// verificaciones ya loggeadas. Se usa una actualización Bayesiana real
// (Beta-Binomial, prior no informativo Beta(1,1)) en vez de un cociente
// simple: con pocos datos, la media Bayesiana se "atempera" hacia 50%
// en vez de sobreinterpretar una racha corta como si fuera señal real.
let tasaAciertosProduccion = null, tasaAciertosSimple = null, totalVerificacionesProduccion = 0,
    credibilidadBaja = null, credibilidadAlta = null;
try {
  const historial = $('Leer Historial Verificaciones').all();
  let aciertos = 0, evaluables = 0;
  for (const item of historial) {
    const r = item.json?.verifResultado;
    if (r === 'acierto' || r === 'error') {
      evaluables++;
      if (r === 'acierto') aciertos++;
    }
  }
  totalVerificacionesProduccion = evaluables;
  if (evaluables > 0) {
    const errores = evaluables - aciertos;
    tasaAciertosSimple = Math.round((aciertos / evaluables) * 100);
    // Beta(1+aciertos, 1+errores) — prior no informativo
    const alphaPost = 1 + aciertos;
    const betaPost = 1 + errores;
    tasaAciertosProduccion = Math.round((alphaPost / (alphaPost + betaPost)) * 100);
    // Aproximación normal al intervalo de credibilidad (suficiente para mostrar en el mail)
    const media = alphaPost / (alphaPost + betaPost);
    const varianza = (alphaPost * betaPost) / (Math.pow(alphaPost + betaPost, 2) * (alphaPost + betaPost + 1));
    const desvio = Math.sqrt(varianza);
    credibilidadBaja = Math.round(Math.max(0, media - 1.96 * desvio) * 100);
    credibilidadAlta = Math.round(Math.min(1, media + 1.96 * desvio) * 100);
  }
} catch (e) {
  tasaAciertosProduccion = null; // todavía no hay suficiente historial logueado
}

// ============================================================
// CÁLCULO DE OPERACIÓN SUGERIDA (Binance)
// ============================================================
const CAPITAL_MAX = 200;
const cantidadEth = +(CAPITAL_MAX / precioActualUSD).toFixed(4);
const tpPrice = +(precioActualUSD * 1.08).toFixed(2);
const slActivador = +(precioActualUSD * 0.90).toFixed(2);
const slLimit = +(slActivador * 0.997).toFixed(2);
const totalUsdt = +(cantidadEth * precioActualUSD).toFixed(2);

const entradaPromedio = +((high24h + low24h) / 2).toFixed(2);
const cantidadPromedio = +(CAPITAL_MAX / entradaPromedio).toFixed(4);
const entradaMargenBajo = +low24h.toFixed(2);
const entradaMargenAlto = +high24h.toFixed(2);
const cantidadMargenBajo = +((CAPITAL_MAX / 2) / entradaMargenBajo).toFixed(4);
const cantidadMargenAlto = +((CAPITAL_MAX / 2) / entradaMargenAlto).toFixed(4);

const datos = {
  precio_usd: precioActualUSD,
  cambio_24h: cambio24h.toFixed(2),
  volumen_24h: volumen24hUSD.toFixed(0),
  market_cap: marketCapAprox.toFixed(0),
  ath: ATH_HISTORICO,
  ath_pct: athPct.toFixed(1),
  cambio_7d: cambio7d.toFixed(2),
  cambio_30d: cambio30d.toFixed(2),
  fecha: new Date().toLocaleDateString('es-AR'),
  fearGreedValue: fg.value,
  fearGreedLabel: fg.value_classification,

  scoreVolatilidad: Math.round(scoreVol),
  scoreTendencia: Math.round(scoreTendencia),
  regimen,
  scoreMomentum: Math.round(scoreMomentum),
  autocorrelacion: +autocorr.toFixed(3),
  scoreATH: Math.round(scoreATH),
  scoreTecnico,

  probabilidadSube,
  colorModelo,
  colorModeloEmoji,
  accuracyModelo: Math.round(accuracyModelo * 100),
  baselineModelo: Math.round(baselineModelo * 100),
  logLossModelo,
  descripcionModelo,
  cantidadFeaturesModelo: pesosModelo.length - 1,

  verifDisponible,
  verifFecha,
  verifProbabilidad,
  verifColor,
  verifPrecioEntonces,
  verifCambioPct,
  verifResultado,
  tasaAciertosProduccion,
  tasaAciertosSimple,
  credibilidadBaja,
  credibilidadAlta,
  totalVerificacionesProduccion,

  capitalMax: CAPITAL_MAX,
  cantidadEth, tpPrice, slActivador, slLimit, totalUsdt,
  high24h: +high24h.toFixed(2), low24h: +low24h.toFixed(2),
  entradaPromedio, cantidadPromedio,
  entradaMargenBajo, cantidadMargenBajo,
  entradaMargenAlto, cantidadMargenAlto
};

return [{ json: datos }];
