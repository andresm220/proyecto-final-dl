# Decisión de producción — qué modelo decide y por qué

**Fecha:** 2026-11-09 03:50 UTC (21:50 hora de Guatemala), después de evaluar H3.
**Quién decide:** el equipo, en el rol del comité de negocio.

## Lo que dijo la regla técnica (congelada en `docs/HIPOTESIS.md`)

| Validación (media de 3 semillas/pares) | AUC-PR | Costo en `u*` |
|---|---:|---:|
| A — LightGBM | 0.8329 | Q270,560 |
| E — ensamble 0.5·A + 0.5·C_K3 | **0.8364** | Q281,720 |

H3 exigía las dos cosas: AUC-PR de E ≥ A **y** costo de E ≤ A. E gana en AUC-PR pero
pierde en costo, así que **H3 no se cumple** y, por la regla, A queda como campeón.
Los artefactos de `artefactos/produccion/` reflejan exactamente eso
(`roles.campeon = A`, `roles.retador = E`) y no se modificaron.

Prueba, evaluada una sola vez y después de congelar todo:

| Prueba (13.4 días, 703 fraudes) | AUC-PR | Recall al bloquear | Costo |
|---|---:|---:|---:|
| A — LightGBM | 0.857 | 97.7 % | Q261,420 |
| E — ensamble | 0.853 | 97.6 % | Q263,460 |

## La decisión: promover E

El enunciado pide un modelo de **Deep Learning** en producción. Con A como campeón, el GRU
solo puntúa en sombra y no participa en ninguna decisión. E sí lo hace: la mitad de cada
puntaje viene del GRU, que es el componente que lee el **orden** de las transacciones
(barajar su ventana le quita 0.125 de AUC-PR).

**Costo de la decisión, cuantificado:** en prueba, E cuesta Q2,040 más que A en 13.4
días, unos **Q4,560 al mes**. Eso es el 0.08 % del ahorro mensual que da cualquiera de los
dos frente a no tener modelo (~Q6 M). En validación la brecha es mayor (Q11,160 de media);
hay que leerla con cuidado, porque en uno de los tres pares E fue más barato que A.

**Cómo se implementa.** No se reentrena ni se recalibra nada. El Dockerfile fija
`PROMOVER_RETADOR=1`, que intercambia roles al arrancar: E decide con su propio
calibrador y su propio umbral (`u* = 2.2 %`), y A pasa a ser el retador en sombra.
`GET /salud` lo declara (`promovido_por_decision_del_equipo: true`) y la versión servida
lleva el sufijo `+promovido`. Quitar esa línea devuelve el servicio a la regla técnica.

**Qué vigila el panel para revisar esta decisión.** Con A en sombra, el panel compara
el costo realizado de ambos sobre el mismo tráfico etiquetado. Si con ≥ 200 fraudes
etiquetados A sigue saliendo más barato, el panel lo marca y la recomendación es volver
a A, o reentrenar el ensamble con un peso distinto, validado con un conjunto nuevo.

## Lo que no afirmamos

- No afirmamos que E sea el mejor modelo: en costo no lo es.
- No buscamos el peso del ensamble en validación (se fijó en 0.5 antes de calcular nada),
  así que el 0.8364 no está inflado por ese ajuste.
- La adenda H3 se escribió después de conocer los resultados individuales de A y C_K3;
  está declarado en `docs/HIPOTESIS.md`.
