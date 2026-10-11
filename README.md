# Predicción de Precios en Steam - MLOps (TC5061)

Este repositorio contiene el código de experimentación desarrollado para predecir el precio de videojuegos en la plataforma Steam mediante la implementación de un modelo `RandomForestRegressor`. El proyecto se encuentra instrumentado con **MLflow Tracking** para garantizar la trazabilidad y reproducibilidad mediante el registro de hiperparámetros, métricas de evaluación y artefactos generados.

## Estructura del Repositorio

- `TC5061_Notebook_Baseline_OjedaLuis.ipynb`: Análisis Exploratorio de Datos (EDA), preprocesamiento y experimentación base.
- `train.py`: Script de entrenamiento parametrizado y registro de experimentos en MLflow.
- `requirements.txt`: Declaración de dependencias y versiones de las librerías necesarias para la correcta ejecución del entorno.

## Requisitos e Instalación

Se recomienda aislar las dependencias utilizando un entorno virtual (venv o conda). Para instalar los paquetes requeridos, ejecute el siguiente comando en la terminal:

```bash
pip install -r requirements.txt
```

## Ejecución de Experimentos

El script `train.py` implementa `argparse` para facilitar la configuración de los hiperparámetros del modelo y garantizar la reproducibilidad mediante la fijación de la semilla aleatoria (`random_seed`) directamente desde la interfaz de línea de comandos (CLI).

Para reproducir los 5 experimentos base de esta investigación, ejecute las siguientes instrucciones:

```bash
python train.py --n_estimators 50 --max_depth 5 --min_samples_split 2 --random_seed 42
python train.py --n_estimators 100 --max_depth 10 --min_samples_split 5 --random_seed 42
python train.py --n_estimators 150 --max_depth 15 --min_samples_split 2 --random_seed 42
python train.py --n_estimators 200 --max_depth 20 --min_samples_split 10 --random_seed 42
python train.py --n_estimators 250 --min_samples_split 5 --random_seed 42
```

## Resultados Destacados

> **Nota de visualización:** La comparativa exhaustiva de los experimentos puede ser consultada iniciando el servidor local de MLflow con el comando `mlflow ui` y accediendo a [http://127.0.0.1:5000](http://127.0.0.1:5000).

Tras el registro sistemático de las configuraciones bajo el experimento `TC5061_Experimentacion_OjedaLuis`, los resultados indican que la profundidad del árbol (`max_depth`) actúa como el hiperparámetro crítico para mitigar el subajuste (underfitting). El modelo alcanzó su convergencia óptima, logrando el mejor balance entre el Error Cuadrático Medio (**RMSE**) y el Coeficiente de Determinación (**R²**), en aquellas iteraciones con mayor ensamble de estimadores. 

El análisis detallado de las métricas y conclusiones derivadas se encuentra documentado en el reporte PDF correspondiente al entregable de la actividad.