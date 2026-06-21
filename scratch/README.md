# Directorio de Scratch: Herramientas de Validación y Diagnóstico de Fase 2

Este directorio contiene herramientas experimentales y scripts auxiliares utilizados durante la fase de fine-tuning y evaluación en el reto **ReXGroundingCT**.

## Estructura Actual

*   **`run_mean_teacher_val_eval.py`**: Script auxiliar para evaluar cuantitativamente checkpoints específicos del entrenamiento Mean Teacher. Genera predicciones 3D mediante sliding window tanto para la red Student como para la red Teacher (EMA), calcula el coeficiente Dice global en el conjunto de validación y los contrasta frente al baseline zero-shot (v1.1).

## Notas de Ejecución

Todos los scripts de este directorio leen la configuración del entorno desde `.env` para heredar las variables de aislamiento de hardware (como `CUDA_VISIBLE_DEVICES`), garantizando un comportamiento predecible y evitando la ocupación involuntaria de recursos en uso por otros usuarios del servidor.
