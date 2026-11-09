# Hipótesis y criterios de éxito — congelados antes de entrenar

**Registrado:** 2026-11-09T02:26:32Z (2026-11-08 20:26, hora de Guatemala).
**Ningún modelo de este documento se había entrenado al escribirlo.** El script
`entrenamiento/entrenar.py` lee este archivo, guarda su SHA-256 en
`artefactos/produccion/metadata.json` y registra la hora de inicio del entrenamiento;
así cualquiera puede comprobar que el criterio no se movió después de ver resultados.

## Punto de partida (Proyecto 1, validación, media de 3 semillas)

| Modelo | AUC-PR val | Nota |
|---|---:|---|
| A — LightGBM sobre agregados (sin orden) | 0.8329 ± 0.0025 | línea base |
| B — GRU, `K = 20` | 0.7138 ± 0.0685 | candidato guardado del P1 |
| C — híbrido GRU + agregados, `K = 20` | 0.6939 ± 0.0311 | |

La curva de recorte del Proyecto 1 (una semilla) mostró que `K = 20` era la peor
configuración del GRU: `K = 3` dio 0.7491 contra 0.6360. El propio README del P1 dejó
escrito que la primera prueba pendiente era reentrenar B y C con `K = 3`.

## Un solo cambio

Se cambia **únicamente** `K: 20 → 3`. Arquitectura, hiperparámetros (`GRU 64`,
dropout 0.3, Adam 1e-3, lotes de 2,048, 30 épocas máx., paciencia 5 sobre AUC-PR de
validación), datos, partición temporal y semillas `(7, 13, 29)` quedan idénticos. Si
algo mejora, se puede atribuir al tamaño de ventana y no a otra cosa.

## Hipótesis

**H1 — Ventana corta.** El GRU con `K = 3` supera al GRU con `K = 20`.

- Métrica: AUC-PR de validación, media de las 3 semillas.
- **Éxito si:** `media(B_K3) − media(B_K20) ≥ 0.03` **y** la peor semilla de `B_K3`
  supera la media de `B_K20` (0.7138). La segunda condición existe porque la
  desviación de `B_K20` entre semillas (0.068) es mayor que la mejora mínima pedida:
  sin ella, una semilla afortunada podría declarar éxito.

**H2 — El orden suma sobre los agregados.** El híbrido C con `K = 3` iguala o supera a
la línea base A.

- Métrica: AUC-PR de validación, media de las 3 semillas.
- **Éxito si:** `media(C_K3) ≥ media(A) − 0.005`.

## Regla de decisión para producción (también congelada)

1. El **campeón** —el modelo cuyo puntaje decide bloquear / revisar / dejar pasar— es
   el de mayor AUC-PR media de validación entre `{A, B_K3, C_K3}`. Si un modelo
   secuencial queda a menos de 0.005 de A, gana A: es más simple, más rápido y más
   fácil de explicar al cliente bloqueado.
2. El mejor de los modelos secuenciales que no salga campeón se despliega como
   **retador en modo sombra**: puntúa cada transacción y su puntaje queda registrado,
   pero no decide nada. Así el comité puede comparar ambos sobre tráfico real sin
   arriesgar dinero.
3. De cada familia se sirve la semilla con mayor AUC-PR de validación. Calibración
   isotónica y umbral `u*` (mínimo costo con FN = Q4,200 y FP = Q180) se ajustan en
   **validación** y se congelan.
4. Test se evalúa **una sola vez**, después de congelar todo lo anterior, y su
   resultado no cambia ninguna decisión: solo se reporta.
