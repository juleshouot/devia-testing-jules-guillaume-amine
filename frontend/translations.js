// translations.js - Gestion multilingue COMPLÈTE
console.log('🌐 Chargement du système de traductions...');

const TRANSLATIONS = {
    en: {
        // Navigation et titres principaux
        title: 'Pandemic Predictions - WHO',
        dashboard: 'Dashboard',
        dashboard_title: 'Pandemic Dashboard',
        predictions: 'Predictions',
        about: 'About',
        ai_predictions: 'AI Predictions',

        // Filtres et sélections
        virus: 'Virus',
        location: 'Location',
        all_viruses: 'All viruses',
        all_locations: 'All locations',
        country: 'Country',
        period: 'Period',
        select_country: 'Select a country...',

        // Métriques principales
        country_population: 'Country Population',
        new_cases_period: 'New Cases (Period)',
        new_deaths_period: 'New Deaths (Period)',
        global_mortality_rate: 'Global Mortality Rate',
        cases_evolution: 'Cases Evolution',

        // Validation des modèles
        model_validation: 'AI Mortality Model Validation',
        transmission_model_validation: 'AI Transmission Model Validation (Rt)',
        validate_model: 'Validate Model',
        validate_rt_model: 'Validate Rt Model',
        validation_results: 'Validation Results',
        transmission_validation_results: 'Transmission Validation Results (Rt)',

        // Métriques de validation
        global_accuracy: 'Global Accuracy',
        rt_future_accuracy: 'Future Rt Accuracy',
        median_error: 'Median error',
        median_error_percent: 'Median error (%)',
        data_points: 'Data points',
        temporal_points: 'Temporal points',
        rt_predictions: 'Rt Predictions',
        prediction_quality: 'Prediction Quality',
        rt_prediction_quality: 'Rt Prediction Quality',

        // Qualité des prédictions
        excellent: 'Excellent',
        excellent_rt: 'Excellent (<15%)',
        good: 'Good',
        good_rt: 'Good (15-30%)',
        fair: 'Fair',
        fair_rt: 'Fair (30-50%)',
        poor: 'Poor',
        poor_rt: 'Poor (>50%)',

        // Métriques des modèles
        model_metrics: 'AI Model Metrics',
        model_performance: 'Prediction model performance',
        model_name: 'Model Name',
        type: 'Type',
        r2_score: 'R² Score',
        loading_metrics: 'Loading metrics...',

        // À propos
        about_description: 'This application was developed to visualize pandemic data and provide predictions based on artificial intelligence models. It uses data from WorldMeter and other sources to feed visualizations and models.',
        disclaimer: 'The predictions provided are for informational purposes only and should not be used to make medical or public health decisions without consulting experts.',
        technologies_used: 'Technologies used:',
        tech_list: 'Django, Django REST framework, Python, Pandas, Scikit-learn, Chart.js, Bootstrap.',
        copyright: '© 2024 Pandemic Predictions Project. All rights reserved.',

        // Messages système
        technical_api: 'Technical API',
        data_visualization: 'Data Visualization'
    },

    fr: {
        // Navigation et titres principaux
        title: 'Prédictions de Pandémies - OMS',
        dashboard: 'Tableau de bord',
        dashboard_title: 'Tableau de bord des pandémies',
        predictions: 'Prédictions',
        about: 'À propos',
        ai_predictions: 'Prédictions IA',

        // Filtres et sélections
        virus: 'Virus',
        location: 'Localisation',
        all_viruses: 'Tous les virus',
        all_locations: 'Toutes les localisations',
        country: 'Pays',
        period: 'Période',
        select_country: 'Sélectionner un pays...',

        // Métriques principales
        country_population: 'Population du pays',
        new_cases_period: 'Nouveaux cas (période)',
        new_deaths_period: 'Nouveaux décès (période)',
        global_mortality_rate: 'Taux de mortalité global',
        cases_evolution: 'Évolution des cas',

        // Validation des modèles
        model_validation: 'Validation du Modèle de Mortalité IA',
        transmission_model_validation: 'Validation du Modèle de Transmission IA (Rt)',
        validate_model: 'Valider le Modèle',
        validate_rt_model: 'Valider le Modèle Rt',
        validation_results: 'Résultats de Validation',
        transmission_validation_results: 'Résultats de Validation Transmission (Rt)',

        // Métriques de validation
        global_accuracy: 'Précision Globale',
        rt_future_accuracy: 'Précision Rt Futur',
        median_error: 'Erreur médiane',
        median_error_percent: 'Erreur médiane (%)',
        data_points: 'Points de données',
        temporal_points: 'Points temporels',
        rt_predictions: 'Prédictions Rt',
        prediction_quality: 'Qualité des Prédictions',
        rt_prediction_quality: 'Qualité des Prédictions Rt',

        // Qualité des prédictions
        excellent: 'Excellentes',
        excellent_rt: 'Excellentes (<15%)',
        good: 'Bonnes',
        good_rt: 'Bonnes (15-30%)',
        fair: 'Correctes',
        fair_rt: 'Correctes (30-50%)',
        poor: 'Mauvaises',
        poor_rt: 'Mauvaises (>50%)',

        // Métriques des modèles
        model_metrics: 'Métriques des Modèles IA',
        model_performance: 'Performance des modèles de prédiction',
        model_name: 'Nom du Modèle',
        type: 'Type',
        r2_score: 'Score R²',
        loading_metrics: 'Chargement des métriques...',

        // À propos
        about_description: 'Cette application a été développée pour visualiser les données de pandémies et fournir des prédictions basées sur des modèles d\'intelligence artificielle. Elle utilise les données de WorldMeter et d\'autres sources pour alimenter les visualisations et les modèles.',
        disclaimer: 'Les prédictions fournies sont à titre indicatif et ne doivent pas être utilisées pour prendre des décisions médicales ou de santé publique sans consultation d\'experts.',
        technologies_used: 'Technologies utilisées :',
        tech_list: 'Django, Django REST framework, Python, Pandas, Scikit-learn, Chart.js, Bootstrap.',
        copyright: '© 2024 Projet de Prédictions de Pandémies. Tous droits réservés.',

        // Messages système
        technical_api: 'API Technique',
        data_visualization: 'Visualisation de Données'
    },

    de: {
        // Navigation et titres principaux
        title: 'Pandemie-Vorhersagen - WHO',
        dashboard: 'Dashboard',
        dashboard_title: 'Pandemie-Dashboard',
        predictions: 'Vorhersagen',
        about: 'Über uns',
        ai_predictions: 'KI-Vorhersagen',

        // Filtres et sélections
        virus: 'Virus',
        location: 'Standort',
        all_viruses: 'Alle Viren',
        all_locations: 'Alle Standorte',
        country: 'Land',
        period: 'Zeitraum',
        select_country: 'Land auswählen...',

        // Métriques principales
        country_population: 'Landesbevölkerung',
        new_cases_period: 'Neue Fälle (Zeitraum)',
        new_deaths_period: 'Neue Todesfälle (Zeitraum)',
        global_mortality_rate: 'Globale Sterblichkeitsrate',
        cases_evolution: 'Entwicklung der Fälle',

        // Validation des modèles
        model_validation: 'KI-Mortalitätsmodell-Validierung',
        transmission_model_validation: 'KI-Übertragungsmodell-Validierung (Rt)',
        validate_model: 'Modell validieren',
        validate_rt_model: 'Rt-Modell validieren',
        validation_results: 'Validierungsergebnisse',
        transmission_validation_results: 'Übertragungsvalidierungsergebnisse (Rt)',

        // Métriques de validation
        global_accuracy: 'Globale Genauigkeit',
        rt_future_accuracy: 'Zukünftige Rt-Genauigkeit',
        median_error: 'Medianer Fehler',
        median_error_percent: 'Medianer Fehler (%)',
        data_points: 'Datenpunkte',
        temporal_points: 'Zeitpunkte',
        rt_predictions: 'Rt-Vorhersagen',
        prediction_quality: 'Vorhersagequalität',
        rt_prediction_quality: 'Rt-Vorhersagequalität',

        // Qualité des prédictions
        excellent: 'Ausgezeichnet',
        excellent_rt: 'Ausgezeichnet (<15%)',
        good: 'Gut',
        good_rt: 'Gut (15-30%)',
        fair: 'Angemessen',
        fair_rt: 'Angemessen (30-50%)',
        poor: 'Schlecht',
        poor_rt: 'Schlecht (>50%)',

        // Métriques des modèles
        model_metrics: 'KI-Modell-Metriken',
        model_performance: 'Leistung der Vorhersagemodelle',
        model_name: 'Modellname',
        type: 'Typ',
        r2_score: 'R²-Score',
        loading_metrics: 'Lade Metriken...',

        // À propos
        about_description: 'Diese Anwendung wurde entwickelt, um Pandemiedaten zu visualisieren und Vorhersagen basierend auf Modellen der künstlichen Intelligenz zu liefern. Sie verwendet Daten von WorldMeter und anderen Quellen, um Visualisierungen und Modelle zu speisen.',
        disclaimer: 'Die bereitgestellten Vorhersagen dienen nur zu Informationszwecken und sollten nicht zur Entscheidungsfindung in medizinischen oder öffentlichen Gesundheitsfragen ohne Rücksprache mit Experten verwendet werden.',
        technologies_used: 'Verwendete Technologien:',
        tech_list: 'Django, Django REST framework, Python, Pandas, Scikit-learn, Chart.js, Bootstrap.',
        copyright: '© 2024 Pandemie-Vorhersagen-Projekt. Alle Rechte vorbehalten.',

        // Messages système
        technical_api: 'Technische API',
        data_visualization: 'Datenvisualisierung'
    },

    it: {
        // Navigation et titres principaux
        title: 'Previsioni Pandemiche - OMS',
        dashboard: 'Cruscotto',
        dashboard_title: 'Cruscotto delle pandemie',
        predictions: 'Previsioni',
        about: 'Chi siamo',
        ai_predictions: 'Previsioni IA',

        // Filtres et sélections
        virus: 'Virus',
        location: 'Posizione',
        all_viruses: 'Tutti i virus',
        all_locations: 'Tutte le posizioni',
        country: 'Paese',
        period: 'Periodo',
        select_country: 'Seleziona un paese...',

        // Métriques principales
        country_population: 'Popolazione del paese',
        new_cases_period: 'Nuovi casi (periodo)',
        new_deaths_period: 'Nuovi decessi (periodo)',
        global_mortality_rate: 'Tasso di mortalità globale',
        cases_evolution: 'Evoluzione dei casi',

        // Validation des modèles
        model_validation: 'Validazione del Modello di Mortalità IA',
        transmission_model_validation: 'Validazione del Modello di Trasmissione IA (Rt)',
        validate_model: 'Convalida Modello',
        validate_rt_model: 'Convalida Modello Rt',
        validation_results: 'Risultati di Validazione',
        transmission_validation_results: 'Risultati di Validazione Trasmissione (Rt)',

        // Métriques de validation
        global_accuracy: 'Precisione Globale',
        rt_future_accuracy: 'Precisione Rt Futuro',
        median_error: 'Errore mediano',
        median_error_percent: 'Errore mediano (%)',
        data_points: 'Punti dati',
        temporal_points: 'Punti temporali',
        rt_predictions: 'Previsioni Rt',
        prediction_quality: 'Qualità delle Previsioni',
        rt_prediction_quality: 'Qualità delle Previsioni Rt',

        // Qualité des prédictions
        excellent: 'Eccellenti',
        excellent_rt: 'Eccellenti (<15%)',
        good: 'Buone',
        good_rt: 'Buone (15-30%)',
        fair: 'Discrete',
        fair_rt: 'Discrete (30-50%)',
        poor: 'Scadenti',
        poor_rt: 'Scadenti (>50%)',

        // Métriques des modèles
        model_metrics: 'Metriche dei Modelli IA',
        model_performance: 'Performance dei modelli di previsione',
        model_name: 'Nome del Modello',
        type: 'Tipo',
        r2_score: 'Punteggio R²',
        loading_metrics: 'Caricamento metriche...',

        // À propos
        about_description: 'Questa applicazione è stata sviluppata per visualizzare i dati delle pandemie e fornire previsioni basate su modelli di intelligenza artificiale. Utilizza dati da WorldMeter e altre fonti per alimentare visualizzazioni e modelli.',
        disclaimer: 'Le previsioni fornite sono solo a scopo informativo e non dovrebbero essere utilizzate per prendere decisioni mediche o di salute pubblica senza consultare esperti.',
        technologies_used: 'Tecnologie utilizzate:',
        tech_list: 'Django, Django REST framework, Python, Pandas, Scikit-learn, Chart.js, Bootstrap.',
        copyright: '© 2024 Progetto Previsioni Pandemiche. Tutti i diritti riservati.',

        // Messages système
        technical_api: 'API Tecnica',
        data_visualization: 'Visualizzazione Dati'
    }
};

// Fonction pour traduire un texte
function translate(key, lang = 'en') {
    if (!TRANSLATIONS[lang]) {
        console.warn(`❌ Langue non supportée: ${lang}`);
        lang = 'en';
    }

    const translation = TRANSLATIONS[lang]?.[key] || TRANSLATIONS.en[key] || key;
    return translation;
}

// Fonction pour appliquer les traductions à la page (version simplifiée, sera remplacée par celle de script.js)
function applyTranslations(lang) {
    console.log(`🔄 Application des traductions: ${lang}`);

    if (!TRANSLATIONS[lang]) {
        console.error(`❌ Traductions non disponibles pour: ${lang}`);
        return;
    }

    let translatedCount = 0;

    document.querySelectorAll('[data-translate]').forEach(element => {
        const key = element.getAttribute('data-translate');
        const translation = translate(key, lang);

        if (translation !== key) {
            element.textContent = translation;
            translatedCount++;
        }
    });

    // Mettre à jour le title de la page
    document.title = translate('title', lang);

    console.log(`✅ ${translatedCount} éléments traduits en ${lang}`);
}

// S'assurer que les traductions sont disponibles globalement
window.TRANSLATIONS = TRANSLATIONS;
window.translate = translate;
window.applyTranslations = applyTranslations;

console.log('✅ Système de traductions chargé avec succès');
console.log('🌐 Langues disponibles:', Object.keys(TRANSLATIONS));