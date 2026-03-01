# Estándar de Puesta en Producción de Estrategias

> Documento de referencia del equipo EaglesEye.
> Cualquier estrategia debe superar este protocolo antes de operar con dinero real.
> Versión: 2.0 — Establecido: 2026-02-28

---

## 1. Filosofía base

Una estrategia no se valida solo por su PnL total. Se valida por su **comportamiento bajo condiciones que no vio durante el desarrollo**. El objetivo es detectar overfitting, fragilidad de régimen y sensibilidad excesiva a costes antes de exponerse al riesgo real.

### Por qué el bloque OOS único es insuficiente

El split estático IS/OOS/Estrés tiene un defecto grave: si el OOS cae en un mes de régimen adverso (agosto, períodos pre-elecciones, alta volatilidad macro), condenas la estrategia por el calendario, no por su lógica. Cambiar manualmente las fechas del OOS para buscar el período que mejor le quede es **p-hacking** y produce autoengaño.

La solución profesional es el **Walk-Forward Analysis (WFA)**: cada mes del historial se evalúa como OOS exactamente una vez. El resultado compuesto de todos esos meses ciegos es la medida honesta del edge real.

---

## 2. Método principal: Walk-Forward Analysis (WFA)

### Funcionamiento

```
IS (6 meses)  →  OOS (1 mes)
    ↓ desliza 1 mes
IS (6 meses)  →  OOS (1 mes)
    ↓ desliza 1 mes
...
```

- **IS deslizante**: ventana de 6 meses que precede a cada OOS. Se usa para calentar el estado de los indicadores (RSI state machine) y heredarlo al OOS. Los parámetros **no se reoptimizen** entre folds — siempre se usan los de producción.
- **OOS independiente**: 1 mes evaluado de forma ciega. Cada mes es OOS exactamente una vez.
- **Resultado compuesto**: suma de los PnL de todos los meses OOS = la verdad estadística.

### Por qué supera al bloque único

| Problema del bloque único | Solución del WFA |
|--------------------------|-----------------|
| Un mes tóxico hunde el OOS | Cada mes es 1/N del total |
| El resultado depende del punto de corte | No hay punto de corte arbitrario |
| No distingue mala racha de overfitting | Si el PF compuesto > 1, el edge existe |
| Pocos trades en OOS (baja significancia) | Suma todos los trades OOS del historial |

### Trampa del p-hacking

No se deben mover las fechas del OOS hasta encontrar el período donde el bot gana. El WFA evalúa **todos los períodos** y reporta el compuesto. Si se ajustan parámetros para mejorar el compuesto WFA, hay que volver a correr con datos frescos no vistos.

---

## 3. Scripts de referencia

| Script | Propósito |
|--------|-----------|
| `research/validate_pivotactivewinner_wfa.py` | **Plantilla WFA principal** — adaptar para cada estrategia |
| `research/validate_pivotactivewinner_production.py` | Split estático 60/20/20 — útil como segunda opinión |

Para adaptar el WFA a una nueva estrategia, modificar en el script:
- Parámetros de producción (TP, SL, filtros, horario)
- Costes por clase de activo (ver Tabla de Costes)
- Path del dataset
- Función `precompute_indicators()` si la estrategia usa indicadores distintos

---

## 4. Proceso paso a paso

### Paso 1 — Preparación del dataset

| Requisito | Mínimo |
|-----------|--------|
| Timeframe de datos crudos | ≤ 1m para estrategias intradía, ≤ 1H para swing |
| Longitud histórica | ≥ 18 meses (preferible ≥ 36 meses) |
| Fuente | Mismo broker/proveedor que operará en real |
| Resample | Al timeframe oficial de la estrategia |

El dataset no se toca entre optimización y validación. Si se descargaron nuevos datos para optimizar, la validación debe hacerse con el historial completo incluyendo el período post-optimización como OOS natural.

### Paso 2 — Precomputar indicadores sin look-ahead

- Los sesgos de marcos superiores (diario, semanal, mensual) se calculan usando únicamente **períodos cerrados**.
- En resampling con `ffill`: `shift(1)` = período cerrado más reciente, `shift(2)` = el anterior. Correcto.
- Prohibido usar el período actual o el cierre del mismo período que se evalúa.
- El estado de la máquina RSI (armado/fuego) se hereda del IS al OOS dentro de cada fold para no perder señales al inicio del mes.

### Paso 3 — Configuración del motor

| Parámetro | Valor estándar |
|-----------|---------------|
| Entrada | `next_open` (vela siguiente a la señal) |
| Colisión TP/SL intrabar | `worst_case` (SL gana) |
| Trades simultáneos | 1 (salvo que la estrategia sea explícitamente multi-posición) |
| Riesgo por trade | 1% de equity por fold |
| Capital inicial por fold | 10,000 (referencia, el PnL absoluto es la métrica) |
| Reoptimización entre folds | **Prohibida** — los parámetros de producción son fijos |

### Paso 4 — Tabla de costes por clase de activo

| Activo | Spread (pts) | Slippage (pts) | Comisión |
|--------|-------------|----------------|----------|
| US30 / NAS100 (CFD Oanda) | 2.5 | 0.7 | 0 |
| XAU/USD (CFD Oanda) | 0.3 | 0.1 | 0 |
| GBP/JPY (CFD Oanda) | 0.02 | 0.01 | 0 |
| BTC/ETH (Binance spot) | 0.05% | 0.05% | 0.1% round-trip |
| BTC/ETH (futuros) | 0.02% | 0.02% | 0.05% round-trip |

Los costes se aplican por lado: entrada suma coste, salida resta coste (longs) o viceversa (shorts).

---

## 5. Criterios de aceptación WFA

### 5.1 Criterios obligatorios (deben cumplirse todos para no bloquear)

| Criterio | Umbral | Justificación |
|----------|--------|---------------|
| PF compuesto OOS | > 1.05 | Edge real por encima de costes |
| PnL compuesto OOS | > 0 | El sistema gana en el agregado de todos los meses ciegos |
| Total trades OOS | ≥ 50 | Significancia estadística mínima |
| Peor mes OOS | > –500 pts absolutos (o > –5% equity) | El peor escenario es controlable |

### 5.2 Criterios de consistencia (se requieren ≥ 2 de 3)

| Criterio | Umbral |
|----------|--------|
| Meses OOS positivos | ≥ 50% |
| WR compuesta OOS | > 35% |
| Ningún trimestre con PF < 0.5 | Ausencia de colapso estructural por régimen |

### 5.3 Tabla de decisión

| Criterios obligatorios | Criterios consistencia | Decisión |
|------------------------|----------------------|----------|
| Todos ✓ | ≥ 2 de 3 ✓ | **Producción — riesgo estándar (1%)** |
| Todos ✓ | 1 de 3 | **Producción controlada (0.5%) + revisión mensual** |
| Todos ✓ | 0 de 3 | **No lista** — revisar lógica o parámetros |
| Alguno ✗ | — | **Bloqueada** — no desplegar |

### 5.4 Producción controlada — definición

- Riesgo por trade: **0.5%** (la mitad del estándar).
- Seguimiento semanal durante el primer mes.
- Revisión formal al cumplir 30 operaciones reales.
- Umbral de revisión obligatoria: drawdown real > 10% del capital asignado.
- Escalar a riesgo estándar solo tras superar una nueva validación WFA con datos post-producción.

---

## 6. Estructura del reporte de validación (entrada en COMMUNICATION_LOG)

```
1. Parámetros exactos de producción usados
2. Dataset: fuente, rango, timeframe base y resampleo
3. Indicadores: verificación de no look-ahead
4. WFA config: IS deslizante, OOS window, paso
5. Costes aplicados: spread, slippage, comisión
6. Tabla de resultados por fold (mes a mes)
7. Métricas compuestas: PF, WR, PnL, meses+/-, peor mes
8. Análisis de consistencia trimestral
9. Estado de criterios obligatorios y de consistencia
10. Veredicto y nivel de riesgo autorizado
11. Firma del agente + aprobación del Usuario
```

---

## 7. Causas de bloqueo automático (no negociable)

- Look-ahead bias detectado en indicadores de marco superior.
- Menos de 50 trades en el agregado OOS.
- PF compuesto OOS ≤ 1.0.
- Parámetros validados distintos a los que irán a producción.
- Dataset distinto al que usará el scanner en tiempo real.
- Reoptimización de parámetros entre folds del WFA.

---

## 8. Registro de estrategias validadas

| Estrategia | Activo | Fecha WFA | PF OOS | Meses+ | Veredicto | Riesgo | Script |
|-----------|--------|----------|--------|--------|-----------|--------|--------|
| PivotActiveWinner v1.1.0 | US30 3m | 2026-02-28 | **1.158** | 4/11 (36%) | 5/7 criterios | **Controlado (0.5%)** | `validate_pivotactivewinner_wfa.py` |

---

## 9. Notas de interpretación

### Meses negativos no son fracaso

Con WFA, es normal tener más meses negativos que positivos si el payoff ratio es > 1. Un sistema con TP=250/SL=150 (ratio 1.67) necesita solo un WR del 38% para ser rentable. Si el PF compuesto > 1, el sistema es viable aunque el 60% de los meses cierren en rojo.

### Dependencia de régimen vs overfitting

Si una estrategia tiene meses malos concentrados en períodos específicos (alta volatilidad macro, rangos laterales estrechos), es dependencia de régimen — una característica del diseño, no overfitting. El WFA lo detecta: si los meses malos son dispersos y no sistemáticos, la estrategia es robusta.

PivotActiveWinner en US30: los meses malos son dispersos (Mar, May, Jul, Sep, Oct, Nov 2025, Ene 2026). No hay colapso trimestral sistemático. El PF compuesto positivo (1.158) confirma edge real.

### Segunda opinión: split estático

El split estático 60/20/20 sigue siendo útil como segunda opinión para visualizar el comportamiento en períodos largos y el régimen más reciente (estrés). No reemplaza el WFA pero complementa el análisis.
