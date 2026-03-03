# 📈 Biblioteca de Indicadores EaglesEye (v2.0)

Este documento contiene la definición matemática y lógica de los indicadores personalizados utilizados en el ecosistema EaglesEye para su implementación en Pine Script y Python (CMD).

---

## 1. 🚀 EMAGap (Tendencia Estructural)
Define la alineación de la tendencia basada en tres medias móviles exponenciales.
- **Componentes:** EMA(34), EMA(89) y EMA(144).
- **Lógica de Estados:**
  - **ALCISTA:** EMA(34) > EMA(89) > EMA(144).
  - **BAJISTA:** EMA(34) < EMA(89) < EMA(144).
  - **NEUTRO:** Cualquier otra combinación (cruce o enredo de medias).

---

## 2. 📍 PivotGap (Filtro Macro)
Mide el desplazamiento del valor intrínseco (tendencia) entre periodos.
- **Cálculo del Pivote (P):** `(High + Low + Close) / 3`.
- **Pivote Actual (PA):** Calculado con los datos del periodo anterior (M-1).
- **Pivote Pasado (PP):** Calculado con los datos de hace dos periodos (M-2).
- **Lógica de Estados (Generalmente en resolución MENSUAL):**
  - **ALCISTA:** PA > PP (El valor intrínseco está subiendo).
  - **BAJISTA:** PA < PP (El valor intrínseco está bajando).

---

## 3. 📉 Slope (Impulso)
Mide la dirección del impulso inmediato de una media móvil específica (ej: Slope34).
- **ALCISTA:** Valor EMA Actual > Valor EMA anterior.
- **BAJISTA:** Valor EMA Actual < Valor EMA anterior.

---

## 4. 🎯 Daily Unhit (Niveles Institucionales)
Identifica soportes/resistencias vírgenes basados en pivotes diarios no testeados.
- **Unhit Bull:** El mínimo (Low) de un día es superior al Pivote del día anterior.
- **Unhit Bear:** El máximo (High) de un día es inferior al Pivote del día anterior.
- **Regla de Secuencia:** Si se producen Unhits en días consecutivos, se mantiene únicamente el nivel del **primer pivote perdido**.
- **Gestión de Stacks:** Almacena hasta **3 niveles alcistas** y **3 niveles bajistas** únicos. La actualización ocurre a las 00:00 UTC.

---

## 5. 📊 RSI State Machine (Wilder)
Oscilador de fuerza relativa con lógica de disparo en dos fases.
- **Cálculo Base:** Método de suavizado de Wilder (EWM con `alpha = 1/N`).
- **Lógica de Estados:**
  1. **ESTADO READY (Armado):** Se activa si el RSI toca niveles de agotamiento extremo (ej: <= 15 para LONG).
  2. **ESTADO TRIGGERED (Disparo):** La señal se genera solo cuando el RSI cruza de vuelta los niveles de recuperación (ej: 25 para LONG).

---

## 6. 📐 Z-Score (Reversión Estadística)
Mide cuántas desviaciones estándar se aleja un valor de su media. Cuantifica si el precio está en un extremo estadístico.

### Configuración Técnica (Sincronización CMD/TradingView)
Para que los resultados en Python coincidan exactamente con TradingView, es obligatorio usar la **Desviación Estándar Muestral**.
- **Fórmula:** `z = (Valor Actual - Media(N)) / DesvEstándar(N)`.
- **Python (Pandas):** Debe usar `ddof=1` en `.std()`.
- **TradingView:** `ta.stdev()` usa `n-1` por defecto.
- **Periodo (N):** 20 para alta frecuencia/scalping; 50-100 para swing trading.

### Lógica de Estados y Filtros
- **SOBREEXTENDIDO ALCISTA:** Z-Score >= +2.5 a +3 (Precio caro).
- **SOBREEXTENDIDO BAJISTA:** Z-Score <= -2.5 a -3 (Precio barato).
- **NEUTRO:** Z-Score entre -1 y +1.
- **Filtro de Tendencia:** Si **EMAGap** es alcista, solo operar extremos bajistas ($Z \leq -2.5$). Si es bajista, solo extremos alcistas ($Z \geq +2.5$).

---

## 7. 🦄 Unicorn Setup (OB + FVG)
Combinación de un bloque de órdenes institucional (Order Block) y una ineficiencia de precio (Fair Value Gap).
- **Order Block (OB):** Última vela cerrada de color contrario antes de un movimiento expansivo que rompe estructura (BMS/MSS).
- **Fair Value Gap (FVG):** Hueco entre la Mecha de la Vela 1 y la Mecha de la Vela 3, donde la Vela 2 es la expansión.
- **Entrada:** En el 50% del FVG ("Consecutive Encroachment") siempre que esté alineado con el OB.
- **Filtro de Sesión:** Operar preferiblemente en 09:00-11:30 y 15:30-18:30 CET.

---

### Notas de Gestión de Riesgo
- **Stop Loss:** En Z-Score, situar en el siguiente nivel de desviación (ej: entrada en 2.5, SL en 3.2). En Unicorn, bajo/sobre el OB.
- **Take Profit:** Para Z-Score, el objetivo primario es el regreso a la media (Z=0).