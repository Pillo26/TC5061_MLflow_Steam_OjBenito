import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import kagglehub
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_preprocess_data():
    """Descarga, limpia y prepara el dataset de Steam usando tu lógica original."""
    # Descargar datos
    path = kagglehub.dataset_download("nikdavis/steam-store-games")
    csv_path = os.path.join(path, "steam.csv")
    df = pd.read_csv(csv_path)

    # Tratamiento de valores faltantes
    df.fillna({'developer': 'Unknown', 'publisher': 'Unknown'}, inplace=True)

    # Corrección de tipos y extracción de características
    df['owners_numeric'] = df['owners'].apply(lambda x: int(x.split('-')[0]))

    # Tratamiento de Outliers (tu lógica del notebook)
    df_clean = df[(df['price'] > 0) & (df['price'] <= 60)].copy()

    # Selección de características para el modelo base
    cols_to_drop = ['appid', 'name', 'release_date', 'developer', 'publisher', 
                    'platforms', 'categories', 'genres', 'steamspy_tags', 'owners']
    df_model = df_clean.drop(columns=cols_to_drop)

    X = df_model.drop(columns=['price'])
    y = df_model['price']
    
    return X, y

def create_feature_importance_plot(model, feature_names, filepath="feature_importance.png"):
    """Crea y guarda un gráfico de importancia de variables como artefacto para MLFlow."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:] # Top 10 variables
    
    plt.figure(figsize=(10, 6))
    plt.title('Top 10 - Importancia de Variables')
    plt.barh(range(len(indices)), importances[indices], color='teal', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importancia Relativa')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def main():
    # 1. Configurar los argumentos de terminal (Parametrización)
    parser = argparse.ArgumentParser(description="Entrenamiento de RandomForest para Steam Dataset")
    parser.add_argument("--n_estimators", type=int, default=100, help="Número de árboles (default: 100)")
    parser.add_argument("--max_depth", type=int, default=None, help="Profundidad máxima del árbol")
    parser.add_argument("--min_samples_split", type=int, default=2, help="Mínimo de muestras para dividir nodo")
    parser.add_argument("--random_seed", type=int, default=42, help="Semilla para reproducibilidad (default: 42)")
    args = parser.parse_args()

    # 2. Configuración de MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("TC5061_Experimentacion_OjedaLuis")

    # Iniciar la corrida en MLflow
    run_name = f"RF_est{args.n_estimators}_depth{args.max_depth}"
    with mlflow.start_run(run_name=run_name):
        
        # 3. Carga y preprocesamiento de datos
        X, y = load_and_preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.random_seed)

        # 4. Registrar Parámetros en MLflow
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("random_seed", args.random_seed)

        # 5. Entrenar el modelo
        rf_model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=args.random_seed
        )
        rf_model.fit(X_train, y_train)

        # 6. Predecir y calcular métricas
        y_pred = rf_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"[{run_name}] MAE: ${mae:.2f} | RMSE: ${rmse:.2f} | R2: {r2:.4f}")

        # 7. Registrar Métricas en MLflow
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # 8. Generar y registrar artefacto visual
        plot_path = "feature_importance.png"
        create_feature_importance_plot(rf_model, X.columns, plot_path)
        mlflow.log_artifact(plot_path)
        os.remove(plot_path) # Eliminar de la carpeta local después de subir a MLflow

        # 9. Registrar el modelo entrenado
        mlflow.sklearn.log_model(
            rf_model, 
            "random_forest_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

if __name__ == "__main__":
    main()