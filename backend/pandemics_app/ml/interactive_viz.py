import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sqlalchemy import create_engine, text
import os
import json
import numpy as np


def generate_interactive_visualizations():
    """Génère des visualisations interactives avec Plotly"""

    # Connexion à la base de données
    def get_engine():
        return create_engine(
            "postgresql+psycopg2://user:guigui@postgres:5432/pandemies",
            connect_args={"options": "-c search_path=pandemics"},
        )

    engine = get_engine()

    # Répertoire de sortie
    output_dir = os.path.join("/django_api", "static", "visualizations")
    os.makedirs(output_dir, exist_ok=True)

    # Données pour les visualisations
    # 1. Évolution des cas
    query = text(
        """
                 SELECT v.name as virus, w.date, SUM(w.new_cases) as new_cases
                 FROM worldmeter w
                          JOIN virus v ON w.virus_id = v.id
                 GROUP BY v.name, w.date
                 ORDER BY w.date
                 """
    )

    cases_df = pd.read_sql(query, engine)
    cases_df["date"] = pd.to_datetime(cases_df["date"])

    # Créer le graphique interactif
    fig = px.line(
        cases_df,
        x="date",
        y="new_cases",
        color="virus",
        title="Évolution des cas par virus",
    )

    # Sauvegarder comme HTML pour intégration directe
    html_path = os.path.join(output_dir, "cases_trend.html")
    fig.write_html(html_path, include_plotlyjs="cdn")

    # Sauvegarder comme JSON pour reconstruction côté client
    json_path = os.path.join(output_dir, "cases_trend.json")
    with open(json_path, "w") as f:
        # Fonction pour convertir des tableaux NumPy en listes
        def convert_numpy_to_list(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_list(i) for i in obj]
            return obj

        # Convertir les données avant de les sérialiser
        fig_dict = fig.to_dict()
        fig_dict_converted = convert_numpy_to_list(fig_dict)
        f.write(json.dumps(fig_dict_converted))

    # 2. Autres visualisations...

    return {"html_files": [html_path], "json_files": [json_path]}
