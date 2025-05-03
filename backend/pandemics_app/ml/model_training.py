import os
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

def get_engine():
    POSTGRES_USER = os.getenv("POSTGRES_USER", "user")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "guigui")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "pandemies")
    return create_engine(
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
        connect_args={"options": "-c search_path=pandemics"}
    )

def load_prepared_data():
    engine = get_engine()
    query = text("""
        SELECT w.*, l.name AS location, v.name AS virus
        FROM worldmeter w
        JOIN location l ON w.location_id = l.id
        JOIN virus v ON w.virus_id = v.id
        ORDER BY w.date ASC
    """)
    df = pd.read_sql(query, engine)
    print("Colonnes dans le DataFrame :", df.columns.tolist())
    if 'new_cases' in df.columns:
        base_cases = df['new_cases'].fillna(0)
    else:
        raise KeyError("Colonne 'new_cases' manquante.")
    df['target_growth_rate'] = base_cases.pct_change(7)
    return df[df['virus'] == 'COVID'], df[df['virus'] == 'Monkeypox']

def prepare_train_test_data(df, target_col='target_growth_rate', test_size=0.2, random_state=42):
    cols_to_exclude = ['id', 'date', 'country', 'location', 'virus', 'virus_id', 'location_id', target_col]
    feature_cols = [col for col in df.columns if col not in cols_to_exclude]

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Données préparées : {X_train.shape[0]} train, {X_test.shape[0]} test")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols



def save_model(model, scaler, feature_cols, model_name, model_type):
    os.makedirs(MODEL_DIR, exist_ok=True)
    safe_name = model_name.replace(" ", "_")
    joblib.dump(model, os.path.join(MODEL_DIR, f"{model_type}_{safe_name}_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"{model_type}_{safe_name}_scaler.joblib"))
    joblib.dump({
        'feature_cols': feature_cols,
        'model_type': model_type,
        'model_name': model_name
    }, os.path.join(MODEL_DIR, f"{model_type}_{safe_name}_metadata.joblib"))
    print(f"✅ Modèle {model_type} sauvegardé ({model_name})")

def train_and_evaluate_models(X_train, X_test, y_train, y_test, model_type):
    print(f"\n=== Entraînement des modèles pour {model_type} ===")
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    param_grids = {
        'Random Forest': {'n_estimators': [50], 'max_depth': [10], 'min_samples_split': [2]},
        'Gradient Boosting': {'n_estimators': [50], 'learning_rate': [0.1], 'max_depth': [3]}
    }
    best_models, model_metrics = {}, {}
    for name, model in models.items():
        print(f"\n→ Entraînement de : {name}")
        if name in param_grids:
            print("  Lancement de GridSearchCV...")
            grid = GridSearchCV(model, param_grids[name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            print(f"  Meilleurs paramètres : {grid.best_params_}")
        else:
            best_model = model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        cv_rmse = np.sqrt(-cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean())
        print(f"  RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.2f} | CV RMSE: {cv_rmse:.2f}")
        best_models[name] = best_model
        model_metrics[name] = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'cv_rmse': cv_rmse}
    best_model_name = min(model_metrics, key=lambda x: model_metrics[x]['rmse'])
    print(f"\n✅ Meilleur modèle pour {model_type} : {best_model_name} (RMSE = {model_metrics[best_model_name]['rmse']:.2f})")
    return best_models, model_metrics, best_model_name

def train_transmission_model():
    print("\n=== Modèle : Taux de transmission ===")
    covid_df, monkeypox_df = load_prepared_data()
    df = pd.concat([covid_df, monkeypox_df])
    df.dropna(subset=['target_growth_rate'], inplace=True)
    df['target_growth_rate'] = df['target_growth_rate'].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_train_test_data(df, 'target_growth_rate')
    models, metrics, best_name = train_and_evaluate_models(X_train, X_test, y_train, y_test, 'transmission')
    save_model(models[best_name], scaler, feature_cols, best_name, 'transmission')
    return metrics

def train_mortality_model():
    print("\n=== Modèle : Taux de mortalité ===")
    covid_df, monkeypox_df = load_prepared_data()
    for df in [covid_df, monkeypox_df]:
        df.sort_values(['location', 'date'], inplace=True)
        df['new_cases'] = df['new_cases'].fillna(0)
        df['new_deaths'] = df['new_deaths'].fillna(0)
        df['cases_7day'] = df['new_cases'].rolling(7).sum()
        df['deaths_7day'] = df['new_deaths'].rolling(7).sum()
        df['cases_7day_lag7'] = df['cases_7day'].shift(7)
        df['deaths_7day_lag7'] = df['deaths_7day'].shift(7)
    covid_df['mortality_ratio'] = covid_df['deaths_7day_lag7'] / covid_df['cases_7day_lag7'].replace(0, 1)
    monkeypox_df['mortality_ratio'] = monkeypox_df['deaths_7day_lag7'] / monkeypox_df['cases_7day_lag7'].replace(0, 1)
    covid_df['mortality_ratio'] = covid_df['mortality_ratio'].clip(0, 1)
    monkeypox_df['mortality_ratio'] = monkeypox_df['mortality_ratio'].clip(0, 1)
    df = pd.concat([covid_df, monkeypox_df]).dropna(subset=['mortality_ratio'])
    X_train, X_test, y_train, y_test, scaler, features = prepare_train_test_data(df, 'mortality_ratio')
    models, metrics, best_name = train_and_evaluate_models(X_train, X_test, y_train, y_test, 'mortality')
    save_model(models[best_name], scaler, features, best_name, 'mortality')
    return metrics

def train_geographical_spread_model():
    print("\n=== Modèle : Propagation géographique ===")
    engine = get_engine()
    df = pd.read_sql(text("""
        SELECT DISTINCT ON (location_id, virus_id) location_id, virus_id, MIN(date) AS date
        FROM worldmeter GROUP BY location_id, virus_id
    """), engine)
    df['week'] = pd.to_datetime(df['date']).dt.isocalendar().week
    df['year'] = pd.to_datetime(df['date']).dt.isocalendar().year
    df['yearweek'] = df['year'].astype(str) + '-' + df['week'].astype(str).str.zfill(2)
    weekly = df.groupby('yearweek').size().reset_index(name='new_locations')
    for i in range(1, 5):
        weekly[f'lag{i}'] = weekly['new_locations'].shift(i)
    weekly['ma2'] = weekly['new_locations'].rolling(2).mean()
    weekly['ma3'] = weekly['new_locations'].rolling(3).mean()
    weekly.dropna(inplace=True)
    X = weekly[[col for col in weekly.columns if col not in ['yearweek', 'new_locations']]]
    y = weekly['new_locations']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    models, metrics, best_name = train_and_evaluate_models(
        scaler.fit_transform(X_train),
        scaler.transform(X_test),
        y_train, y_test, 'geographical_spread'
    )
    save_model(models[best_name], scaler, X.columns.tolist(), best_name, 'geographical_spread')
    return metrics

def train_models():
    print("=== Début de l'entraînement des modèles ===")
    transmission = train_transmission_model()
    mortality = train_mortality_model()
    geographical = train_geographical_spread_model()
    print("\n✅ Entraînement terminé.")
    return transmission, mortality, geographical

if __name__ == "__main__":
    train_models()
