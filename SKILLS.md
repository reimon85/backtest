# 🛠️ Dominio de Backtesting y Robustez Estadística

Para ser el "Rey del Backtest", no basta con que la curva de beneficios suba. Es necesario demostrar que el sistema es estadísticamente resiliente y que los resultados no son fruto del azar o del sobreajuste.

## 1. Conceptos Críticos y Sesgos (Biases)

| Sesgo | Descripción | Mitigación |
| :--- | :--- | :--- |
| **Look-ahead** (Mirada al futuro) | Usar información que no estaba disponible en el momento del trade. | Usar datos *Point-in-time* (estrictamente históricos). |
| **Supervivencia** (Survivorship) | Testear solo en activos que existen hoy (ignorando los que quebraron). | Incluir activos delistados o desaparecidos en el dataset. |
| **Sobreajuste** (Overfitting) | Forzar la estrategia para que encaje perfectamente en el pasado. | Pruebas fuera de muestra (*Out-of-Sample*) y Monte Carlo. |
| **Selección** (Cherry-picking) | Elegir solo los periodos o activos donde la estrategia brilló. | Pre-registro de la estrategia y tests en mercados diversos. |
| **Transacción** | Ignorar spreads, comisiones y deslizamientos (*Slippage*). | Modelos de costes realistas y conservadores. |

## 2. Estructura de un Backtest Profesional (Protocolo de Blindaje)

Para evitar el autoengaño, la separación de datos debe ser sagrada y unidireccional:

1.  **Conjunto de Entrenamiento (In-Sample):** Donde se desarrolla la lógica y se optimizan los parámetros iniciales.
2.  **Conjunto de Validación:** Donde se seleccionan los mejores parámetros sin mirar el test final.
3.  **Conjunto de Prueba (Out-of-Sample):** El examen final. Se toca **una sola vez**. Si falla aquí, la estrategia se descarta, no se "ajusta".



## 3. Análisis Walk-Forward (Ventanas Móviles)
Simula la re-optimización periódica que un trader haría en la vida real.

```text
Ventana 1: [Entrenar──────][Testear]
Ventana 2:     [Entrenar──────][Testear]
Ventana 3:         [Entrenar──────][Testear]
Ventana 4:             [Entrenar──────][Testear]
                                     ─────▶ Tiempo
4. Profundización: Simulación de Monte Carlo (El Test de Estrés)
El pasado es solo una de las infinitas formas en que el mercado pudo haberse movido. La Simulación de Monte Carlo evalúa si tu estrategia sobreviviría si el orden de los eventos cambiara ligeramente.

¿Cómo funciona?
Se toman todos los trades generados por el backtest y se "barajan" aleatoriamente miles de veces (normalmente 1.000 o 10.000 iteraciones) para crear miles de curvas de capital distintas basadas en los mismos datos.

Métricas de Evaluación de Monte Carlo:
Probabilidad de Ruina (Risk of Ruin): ¿En cuántas de las 1.000 simulaciones la cuenta llega a cero o pierde un porcentaje inaceptable (ej. 30%)? Un sistema robusto debe tener un Riesgo de Ruina cercano al 0%.

Drawdown de Confianza (95% CI): Si el 95% de las simulaciones muestran un Drawdown máximo de, digamos, el 15%, puedes esperar con alta confianza que tu riesgo real en el futuro ronde ese número.

Expectativa Mediana vs. Máxima: Si tu backtest original es mucho mejor que la mediana de Monte Carlo, probablemente tuviste una racha de suerte en el orden de los trades originales.

Cuándo confiar en el Edge:
Si la curva de capital del backtest está cerca de la mediana de las simulaciones.

Si el peor escenario de las simulaciones sigue siendo aceptable para tu psicología y capital.

Si al eliminar el 5% de los mejores trades, el sistema sigue siendo rentable.

5. Análisis de Sensibilidad (3D Optimization)
Un sistema "noble" debe ser estable. Si cambias un parámetro (ej. un RSI de 14 a 15) y la rentabilidad se desploma, el sistema es frágil.

El Objetivo: Buscar "mesetas de rentabilidad". Áreas donde un rango de parámetros da resultados positivos consistentes. Evita los "picos aislados" de beneficio, que son puro ruido estadístico.