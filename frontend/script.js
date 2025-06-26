// Script complet pour le tableau de bord des pandémies avec données agrégées
// Variable globale pour tracker si les données sont chargées
let dataLoaded = {
    viruses: false,
    locations: false
};

document.addEventListener('DOMContentLoaded', function() {
  console.log('[INFO] Script de tableau de bord des pandémies démarré');

  // =================== Configuration et constantes ===================

  // Périodes définies pour chaque virus
  const VIRUS_PERIODS = {
    'Monkeypox': {
      start: '2022-05-01',
      end: '2023-05-05'
    },
    'COVID': {
      start: '2020-02-15',
      end: '2022-05-14'
    }
  };

  // Cartographie des couleurs
  const CHART_COLORS = {
    totalCases: '#0077b6',
    totalDeaths: '#d00000',
    newCases: '#20c997',
    newDeaths: '#ff9500',
    casesPerMillion: '#7209b7',
    deathsPerMillion: '#f72585'
  };

  // =================== Fonctions utilitaires ===================

  // Fonction de journalisation améliorée
  function log(level, message, data) {
    const timestamp = new Date().toLocaleTimeString();

    const CSS_STYLES = {
      INFO: 'color: #0077b6; font-weight: normal;',
      DEBUG: 'color: #888; font-weight: normal;',
      WARN: 'color: #ff9500; font-weight: bold;',
      ERROR: 'color: #d00000; font-weight: bold;',
      SUCCESS: 'color: #028a0f; font-weight: bold;'
    };

    const style = CSS_STYLES[level] || '';

    if (data !== undefined) {
      console.log(`%c[${level}] ${timestamp} - ${message}`, style, data);
    } else {
      console.log(`%c[${level}] ${timestamp} - ${message}`, style);
    }
  }

  // Fonction pour afficher un message d'état dans l'interface
  function showStatus(message, type = 'info') {
    // Vérifier si l'élément existe déjà
    let statusEl = document.getElementById('dashboard-status');

    // Créer l'élément s'il n'existe pas
    if (!statusEl) {
      statusEl = document.createElement('div');
      statusEl.id = 'dashboard-status';
      statusEl.style.margin = '15px 0';
      statusEl.style.padding = '10px 15px';
      statusEl.style.borderRadius = '5px';
      statusEl.style.fontWeight = 'bold';

      // Trouver un bon endroit pour l'insérer
      const mainContent = document.querySelector('#main-content') || document.body;
      const firstSection = mainContent.querySelector('section') || mainContent.firstChild;
      mainContent.insertBefore(statusEl, firstSection);
    }

    // Définir le style selon le type
    switch(type) {
      case 'error':
        statusEl.style.backgroundColor = '#f8d7da';
        statusEl.style.color = '#721c24';
        statusEl.style.borderColor = '#f5c6cb';
        break;
      case 'success':
        statusEl.style.backgroundColor = '#d4edda';
        statusEl.style.color = '#155724';
        statusEl.style.borderColor = '#c3e6cb';
        break;
      case 'warning':
        statusEl.style.backgroundColor = '#fff3cd';
        statusEl.style.color = '#856404';
        statusEl.style.borderColor = '#ffeeba';
        break;
      default: // info
        statusEl.style.backgroundColor = '#d1ecf1';
        statusEl.style.color = '#0c5460';
        statusEl.style.borderColor = '#bee5eb';
    }

    // Mettre à jour le contenu
    statusEl.innerHTML = message;

    // Rendre visible
    statusEl.style.display = 'block';
  }

  // Fonction pour formater une date en français
  function formatDateFr(date) {
    if (typeof date === 'string') {
      date = new Date(date);
    }
    return date.toLocaleDateString('fr-FR', { day: 'numeric', month: 'long', year: 'numeric' });
  }

  // Fonction pour formater une date en format court
  function formatShortDate(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleDateString('fr-FR', { day: 'numeric', month: 'short' });
  }

  // Fonction pour formater les valeurs numériques des métriques
  function formatMetricValue(value) {
    if (value === undefined || value === null) return 'N/A';

    // Gérer les valeurs scientifiques (très petites ou très grandes)
    if (typeof value === 'number') {
      if (Math.abs(value) < 0.001 || Math.abs(value) > 999999) {
        return value.toExponential(4);
      }
      return value.toFixed(4);
    }

    return value;
  }

  // =================== Éléments du DOM ===================

  // Récupérer les éléments de l'interface
  const virusSelect = document.getElementById('virus-select');
  const locationSelect = document.getElementById('location-select');
  const totalCasesElem = document.getElementById('total-cases');
  const totalDeathsElem = document.getElementById('total-deaths');
  const newCasesElem = document.getElementById('new-cases');
  const newDeathsElem = document.getElementById('new-deaths');

  // Vérifier si les éléments existent
  if (!virusSelect || !locationSelect) {
    log('ERROR', 'Un ou plusieurs sélecteurs non trouvés');
    showStatus('Erreur: Impossible de trouver les éléments de sélection nécessaires.', 'error');
    return;
  }

  log('INFO', 'Éléments de l\'interface trouvés');

  // Récupérer ou créer le conteneur de date personnalisé
  let dateContainer = document.getElementById('custom-date-selector');
  if (!dateContainer) {
    dateContainer = document.querySelector('.filter-section') ||
                    document.createElement('div');
    if (!dateContainer.classList.contains('filter-section')) {
      dateContainer.className = 'filter-section';
      dateContainer.style.marginTop = '20px';
      dateContainer.style.padding = '15px';
      dateContainer.style.backgroundColor = '#f8f9fa';
      dateContainer.style.borderRadius = '5px';
      dateContainer.style.border = '1px solid #e9ecef';

      // Trouver un bon endroit pour l'insérer
      const mainContent = document.querySelector('#main-content') || document.body;
      const firstSection = mainContent.querySelector('section') || mainContent.firstChild;
      mainContent.insertBefore(dateContainer, firstSection);
    }
  }

  // =================== Fonctions pour charger les données ===================

  // Fonction pour charger les virus depuis l'API
 function loadViruses() {
    log('INFO', 'Chargement des virus...');
    fetch('http://localhost:8001/api/viruses/')
        .then(response => {
            if (!response.ok) {
                throw new Error(`Erreur HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            log('INFO', 'Virus chargés:', data);
            populateSelect(virusSelect, data.results || [], 'name', 'id');

            // Mettre à jour le sélecteur de dates après le chargement
            updateDateSelector();

            // Remplir également les sélecteurs pour les prédictions
            const predVirusSelect = document.getElementById('pred-virus-select');
            const geoPredVirusSelect = document.getElementById('geo-pred-virus-select');

            if (predVirusSelect) {
                populateSelect(predVirusSelect, data.results || [], 'name', 'id');
            }

            if (geoPredVirusSelect) {
                populateSelect(geoPredVirusSelect, data.results || [], 'name', 'id');
            }

            // Marquer les virus comme chargés
            dataLoaded.viruses = true;
            checkAndInitializePredictions();
        })
        .catch(error => {
            log('ERROR', 'Erreur lors du chargement des virus:', error);
            showStatus('Erreur lors du chargement des virus. Veuillez réessayer.', 'error');
        });
}

 function loadAllLocations() {
    log('INFO', 'Chargement de toutes les localisations...');
    showStatus('Chargement des localisations...', 'info');

    let allLocations = [];

    function loadPage(url) {
        return fetch(url)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Erreur HTTP: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                allLocations = allLocations.concat(data.results || []);
                showStatus(`Chargement des localisations... ${allLocations.length}/${data.count}`, 'info');

                if (data.next) {
                    return loadPage(data.next);
                }
                return allLocations;
            });
    }

    loadPage('http://localhost:8001/api/locations/')
        .then(locations => {
            log('INFO', `Toutes les localisations chargées (${locations.length})`, locations);
            showStatus(`${locations.length} localisations chargées avec succès.`, 'success');

            populateSelect(locationSelect, locations, 'name', 'id');

            const predLocationSelect = document.getElementById('pred-location-select');
            if (predLocationSelect) {
                populateSelect(predLocationSelect, locations, 'name', 'id');
            }

            // Marquer les localisations comme chargées
            dataLoaded.locations = true;
            checkAndInitializePredictions();
        })
        .catch(error => {
            log('ERROR', 'Erreur lors du chargement des localisations:', error);
            showStatus('Erreur lors du chargement des localisations. Veuillez réessayer.', 'error');
        });
}

  // Fonction pour remplir un select avec des options
function populateSelect(selectElement, items, textProperty, valueProperty) {
    if (!selectElement) {
        log('ERROR', 'Élément select non trouvé');
        return;
    }

    // Conserver l'option "Tous" ou "all"
    const allOption = Array.from(selectElement.options).find(option =>
        option.value === 'all' || option.textContent.toLowerCase().includes('tous')
    );

    // Vider le select
    selectElement.innerHTML = '';

    // Remettre l'option "Tous" si elle existait
    if (allOption) {
        selectElement.appendChild(allOption);
    } else {
        // Créer une option "Tous" par défaut si elle n'existe pas
        const defaultOption = document.createElement('option');
        defaultOption.value = 'all';
        defaultOption.textContent = 'Tous';
        selectElement.appendChild(defaultOption);
    }

    // Trier les éléments par nom
    const sortedItems = [...items].sort((a, b) => {
        const aText = a[textProperty] || '';
        const bText = b[textProperty] || '';
        return aText.localeCompare(bText);
    });

    // Ajouter les nouvelles options
    sortedItems.forEach(item => {
        if (item[textProperty] && item[valueProperty]) {
            const option = document.createElement('option');
            option.value = item[valueProperty]; // Utiliser l'ID comme valeur
            option.textContent = item[textProperty]; // Utiliser le nom comme texte
            selectElement.appendChild(option);
        }
    });

    log('DEBUG', `${selectElement.id} rempli avec ${items.length} options`);
}
  // =================== Fonctions pour l'interface utilisateur ===================

  // Fonction pour mettre à jour le sélecteur de dates
  function updateDateSelector() {
    const virusName = virusSelect.options[virusSelect.selectedIndex].text;
    const period = VIRUS_PERIODS[virusName];

    if (!period) {
      return;
    }

    // Obtenir ou créer la section Période
    let periodSection = document.querySelector('.periode-section');
    if (!periodSection) {
      periodSection = document.createElement('div');
      periodSection.className = 'periode-section';
      dateContainer.appendChild(periodSection);
    }

    // Mise à jour du contenu de la section Période
    periodSection.innerHTML = `
      <h4 style="margin-top: 0;">Période (${virusName})</h4>
      <p style="margin-bottom: 15px;">
        Période disponible: ${formatDateFr(period.start)} - ${formatDateFr(period.end)}
      </p>
      <div style="display: flex; gap: 15px; margin-bottom: 15px;">
        <div style="flex: 1;">
          <label for="start-date">Date de début:</label>
          <input type="date" id="start-date" min="${period.start}" max="${period.end}" value="${period.start}" style="width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; margin-top: 5px;">
        </div>
        <div style="flex: 1;">
          <label for="end-date">Date de fin:</label>
          <input type="date" id="end-date" min="${period.start}" max="${period.end}" value="${period.end}" style="width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; margin-top: 5px;">
        </div>
      </div>
      <button id="search-button" style="background-color: #007bff; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;">Rechercher</button>
    `;

    // Ajouter les écouteurs d'événements
    const startDateInput = document.getElementById('start-date');
    const endDateInput = document.getElementById('end-date');
    const searchButton = document.getElementById('search-button');

    if (startDateInput && endDateInput) {
      startDateInput.addEventListener('change', function() {
        if (new Date(this.value) > new Date(endDateInput.value)) {
          endDateInput.value = this.value;
        }
      });

      endDateInput.addEventListener('change', function() {
        if (new Date(this.value) < new Date(startDateInput.value)) {
          startDateInput.value = this.value;
        }
      });
    }

    if (searchButton) {
      searchButton.addEventListener('click', function() {
        // Ajouter un effet visuel au bouton
        this.disabled = true;
        this.textContent = 'Recherche en cours...';
        this.style.backgroundColor = '#6c757d';

        // Appeler la fonction de recherche
        fetchAggregatedData().finally(() => {
          // Restaurer le bouton après la recherche
          this.disabled = false;
          this.textContent = 'Rechercher';
          this.style.backgroundColor = '#007bff';
        });
      });
    }
  }

  // =================== Fonctions pour récupérer et traiter les données ===================

  // Fonction pour récupérer les données agrégées
  function fetchAggregatedData() {
    showStatus('Récupération des données agrégées...', 'info');

    const virusName = virusSelect.options[virusSelect.selectedIndex].text;
    const locationName = locationSelect.options[locationSelect.selectedIndex].text;
    const startDateInput = document.getElementById('start-date');
    const endDateInput = document.getElementById('end-date');

    const startDate = startDateInput ? startDateInput.value : null;
    const endDate = endDateInput ? endDateInput.value : null;

    // Créer les paramètres de requête
    let params = new URLSearchParams();

    if (virusName && virusName !== 'all') {
      params.append('virus_name', virusName);
    }

    if (locationName && locationName !== 'all') {
      params.append('location_name', locationName);
    }

    if (startDate) {
      params.append('start_date', startDate);
    }

    if (endDate) {
      params.append('end_date', endDate);
    }

    const url = `http://localhost:8001/api/aggregated-data/?${params.toString()}`;
    log('INFO', 'Récupération des données agrégées:', url);

    return fetch(url)
      .then(response => {
        if (!response.ok) {
          throw new Error(`Erreur HTTP: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        log('INFO', 'Données agrégées récupérées:', data);
        showStatus(`Données agrégées récupérées avec succès (${data.length} jours)`, 'success');

        displayAggregatedData(data);
        return data;
      })
      .catch(error => {
        log('ERROR', 'Erreur lors de la récupération des données agrégées:', error);
        showStatus(`Erreur lors de la récupération des données: ${error.message}`, 'error');
        throw error;
      });
  }

  // Fonction pour afficher les données agrégées
 function displayAggregatedData(data) {
  if (!data || !Array.isArray(data) || data.length === 0) {
    log('WARN', 'Aucune donnée agrégée à afficher');
    showStatus('Aucune donnée trouvée pour les critères sélectionnés.', 'warning');
    return;
  }

  // Calculer les totaux
  const stats = calculateAggregatedStats(data);

  // NOUVEAU: Récupérer les données de localisation pour la population
  const locationName = locationSelect.options[locationSelect.selectedIndex].text;
  if (locationName && locationName !== 'all') {
    fetchLocationData(locationName).then(locationData => {
      updateCards(stats, locationData);
    });
  } else {
    updateCards(stats);
  }

  // Mettre à jour le tableau et graphiques
  updateDataTable(data);
  updateCharts(data);

  log('INFO', 'Affichage mis à jour avec les données agrégées');
}

  function calculateAggregatedStats(data) {
  const stats = {
    totalCases: 0,
    totalDeaths: 0,
    newCases: 0,
    newDeaths: 0,
    avgCasesPerMillion: 0,
    avgDeathsPerMillion: 0,
    globalMortalityRate: 0,
    count: data.length
  };

  if (data.length > 0) {
    // Calculer les sommes pour les nouveaux cas et décès
    data.forEach(item => {
      stats.newCases += Number(item.sum_new_cases || 0);
      stats.newDeaths += Number(item.sum_new_deaths || 0);

      // Pour les statistiques par million, ne prendre en compte que les valeurs non nulles
      if (item.avg_new_cases_per_million !== null) {
        stats.avgCasesPerMillion += Number(item.avg_new_cases_per_million || 0);
      }
      if (item.avg_new_deaths_per_million !== null) {
        stats.avgDeathsPerMillion += Number(item.avg_new_deaths_per_million || 0);
      }

      // Pour les cas et décès totaux, prendre la valeur la plus récente non nulle
      // (les résultats sont typiquement triés par date)
      if (item.total_cases !== null && item.total_cases !== undefined) {
        stats.totalCases = Number(item.total_cases);
      }
      if (item.total_deaths !== null && item.total_deaths !== undefined) {
        stats.totalDeaths = Number(item.total_deaths);
      }
    });

    // Si nous n'avons pas trouvé de total, essayons d'extraire de la dernière entrée
    const lastItem = data[data.length - 1];
    if ((stats.totalCases === 0 || stats.totalCases === undefined) && lastItem) {
      stats.totalCases = Number(lastItem.total_cases || 0);
    }
    if ((stats.totalDeaths === 0 || stats.totalDeaths === undefined) && lastItem) {
      stats.totalDeaths = Number(lastItem.total_deaths || 0);
    }

    // Calculer le taux de mortalité global (nouveaux décès / nouveaux cas) * 100
    if (stats.newCases > 0) {
      stats.globalMortalityRate = (stats.newDeaths / stats.newCases) * 100;
    } else {
      stats.globalMortalityRate = 0;
    }

    // Calculer les moyennes pour les valeurs par million
    const nonNullCasesPerMillion = data.filter(item => item.avg_new_cases_per_million !== null).length;
    const nonNullDeathsPerMillion = data.filter(item => item.avg_new_deaths_per_million !== null).length;

    if (nonNullCasesPerMillion > 0) {
      stats.avgCasesPerMillion = stats.avgCasesPerMillion / nonNullCasesPerMillion;
    }
    if (nonNullDeathsPerMillion > 0) {
      stats.avgDeathsPerMillion = stats.avgDeathsPerMillion / nonNullDeathsPerMillion;
    }
  }

  log('INFO', 'Statistiques calculées à partir des données agrégées:', stats);
  return stats;
}
// Fonction pour vérifier si toutes les données sont chargées et initialiser les prédictions
function checkAndInitializePredictions() {
    if (dataLoaded.viruses && dataLoaded.locations) {
        console.log("=== Toutes les données sont chargées, initialisation des prédictions ===");

        setTimeout(() => {
            const predVirusSelect = document.getElementById('pred-virus-select');
            const predLocationSelect = document.getElementById('pred-location-select');
            const geoPredVirusSelect = document.getElementById('geo-pred-virus-select');

            // Sélectionner des valeurs par défaut pour les prédictions
            if (predVirusSelect && predVirusSelect.options.length > 1) {
                for (let i = 1; i < predVirusSelect.options.length; i++) {
                    if (predVirusSelect.options[i] && predVirusSelect.options[i].textContent === 'COVID') {
                        predVirusSelect.selectedIndex = i;
                        console.log(`COVID sélectionné à l'index ${i}`);
                        break;
                    }
                }
            }

            if (predLocationSelect && predLocationSelect.options.length > 1) {
                for (let i = 1; i < predLocationSelect.options.length; i++) {
                    if (predLocationSelect.options[i] && predLocationSelect.options[i].textContent === 'France') {
                        predLocationSelect.selectedIndex = i;
                        console.log(`France sélectionnée à l'index ${i}`);
                        break;
                    }
                }
            }

            if (geoPredVirusSelect && geoPredVirusSelect.options.length > 1) {
                for (let i = 1; i < geoPredVirusSelect.options.length; i++) {
                    if (geoPredVirusSelect.options[i] && geoPredVirusSelect.options[i].textContent === 'COVID') {
                        geoPredVirusSelect.selectedIndex = i;
                        console.log(`COVID géo sélectionné à l'index ${i}`);
                        break;
                    }
                }
            }

            // Charger les prédictions
            if (predVirusSelect && predLocationSelect &&
                predVirusSelect.selectedIndex > 0 && predLocationSelect.selectedIndex > 0) {
                console.log("Chargement des prédictions transmission/mortalité...");
                loadPredictions();
            }

            if (geoPredVirusSelect && geoPredVirusSelect.selectedIndex > 0) {
                console.log("Chargement des prédictions géographiques...");
                loadGeoPredictions();
            }
        }, 500); // Court délai pour s'assurer que le DOM est bien mis à jour
    }
}
// Fonction pour mettre à jour les cartes avec population
function updateCards(stats, locationData = null) {
  // Taux de mortalité global
  if (totalCasesElem) totalCasesElem.textContent = (stats.globalMortalityRate || 0).toFixed(2) + '%';

  // Population du pays
  if (totalDeathsElem) {
    if (locationData && locationData.population) {
      const population = Number(locationData.population);
      totalDeathsElem.textContent = population.toLocaleString();
      log('INFO', `Population affichée: ${population.toLocaleString()}`);
    } else {
      totalDeathsElem.textContent = 'N/A';
      log('WARN', 'Aucune donnée de population disponible');
    }
  }

  // Nouveaux cas et décès
  if (newCasesElem) newCasesElem.textContent = (stats.newCases || 0).toLocaleString();
  if (newDeathsElem) newDeathsElem.textContent = (stats.newDeaths || 0).toLocaleString();

  // Mettre à jour les étiquettes si elles existent
  updateCardLabels();
}

// Fonction pour mettre à jour les étiquettes des cartes
function updateCardLabels() {
  const cards = document.querySelectorAll('.stat-card');

  cards.forEach((card, index) => {
    const label = card.querySelector('.label');
    if (label) {
      switch(index) {
        case 0:
          label.textContent = 'Population du pays';
          break;
        case 1:
          label.textContent = 'Nouveaux cas (période)';
          break;
        case 2:
          label.textContent = 'Nouveaux décès (période)';
          break;
        case 3:
          label.textContent = 'Taux de mortalité global';
          break;
      }
    }
  });
}
  // Fonction pour mettre à jour le tableau de données
  function updateDataTable(data) {
    const tableBody = document.getElementById('worldmeter-table-body');
    if (!tableBody) {
      return; // Tableau non trouvé, ne rien faire
    }

    if (!data || data.length === 0) {
      tableBody.innerHTML = '<tr><td colspan="7">Aucune donnée disponible pour les filtres sélectionnés.</td></tr>';
      return;
    }

    let html = '';

    data.forEach(item => {
      html += `
        <tr>
          <td>${formatShortDate(item.date)}</td>
          <td>${Number(item.sum_new_cases || 0).toLocaleString()}</td>
          <td>${Number(item.sum_new_deaths || 0).toLocaleString()}</td>
          <td>${Number(item.avg_new_cases_per_million || 0).toFixed(2)}</td>
          <td>${Number(item.avg_new_deaths_per_million || 0).toFixed(2)}</td>
        </tr>
      `;
    });

    tableBody.innerHTML = html;
  }

  // =================== Fonctions pour les graphiques ===================

  // Fonction pour mettre à jour les graphiques
  function updateCharts(data) {
    // Vérifier si Chart.js est disponible
    if (typeof Chart === 'undefined') {
      log('WARN', 'Chart.js non disponible, impossible de mettre à jour les graphiques');
      return;
    }

    // Mettre à jour le graphique des cas
    updateCasesChart(data);

    // Mettre à jour le graphique des cas par million
    updatePerMillionChart(data);
  }

  // Fonction pour mettre à jour le graphique des cas
  function updateCasesChart(data) {
    const casesChartCanvas = document.getElementById('cases-chart');
    if (!casesChartCanvas) {
      return; // Graphique non trouvé
    }

    // Préparer les données pour le graphique
    const dates = data.map(item => formatShortDate(item.date));
    const newCases = data.map(item => Number(item.sum_new_cases || 0));
    const newDeaths = data.map(item => Number(item.sum_new_deaths || 0));

    // Détruire le graphique existant s'il y en a un
    if (window.casesChart instanceof Chart) {
      window.casesChart.destroy();
    }

    // Créer un nouveau graphique
    window.casesChart = new Chart(casesChartCanvas, {
      type: 'bar',
      data: {
        labels: dates,
        datasets: [
          {
            label: 'Nouveaux cas',
            data: newCases,
            backgroundColor: CHART_COLORS.newCases,
            borderColor: CHART_COLORS.newCases,
            borderWidth: 1
          },
          {
            label: 'Nouveaux décès',
            data: newDeaths,
            backgroundColor: CHART_COLORS.newDeaths,
            borderColor: CHART_COLORS.newDeaths,
            borderWidth: 1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
          },
          tooltip: {
            mode: 'index',
            intersect: false
          },
          title: {
            display: true,
            text: 'Évolution des nouveaux cas et décès'
          }
        },
        scales: {
          x: {
            ticks: {
              maxRotation: 45,
              minRotation: 45
            }
          },
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Nombre'
            }
          }
        }
      }
    });
  }

  // Fonction pour mettre à jour le graphique des cas par million
  function updatePerMillionChart(data) {
    const perMillionChartCanvas = document.getElementById('location-chart'); // Réutiliser le canvas existant
    if (!perMillionChartCanvas) {
      return; // Graphique non trouvé
    }

    // Préparer les données pour le graphique
    const dates = data.map(item => formatShortDate(item.date));
    const casesPerMillion = data.map(item => Number(item.avg_new_cases_per_million || 0));
    const deathsPerMillion = data.map(item => Number(item.avg_new_deaths_per_million || 0));

    // Détruire le graphique existant s'il y en a un
    if (window.perMillionChart instanceof Chart) {
      window.perMillionChart.destroy();
    }

    // Créer un nouveau graphique
    window.perMillionChart = new Chart(perMillionChartCanvas, {
      type: 'line',
      data: {
        labels: dates,
        datasets: [
          {
            label: 'Nouveaux cas par million',
            data: casesPerMillion,
            backgroundColor: 'rgba(114, 9, 183, 0.1)',
            borderColor: CHART_COLORS.casesPerMillion,
            borderWidth: 2,
            fill: true,
            tension: 0.4
          },
          {
            label: 'Nouveaux décès par million',
            data: deathsPerMillion,
            backgroundColor: 'rgba(247, 37, 133, 0.1)',
            borderColor: CHART_COLORS.deathsPerMillion,
            borderWidth: 2,
            fill: true,
            tension: 0.4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
          },
          tooltip: {
            mode: 'index',
            intersect: false
          },
          title: {
            display: true,
            text: 'Évolution des cas et décès par million'
          }
        },
        scales: {
          x: {
            ticks: {
              maxRotation: 45,
              minRotation: 45
            }
          },
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Par million d\'habitants'
            }
          }
        }
      }
    });
  }

  // =================== Fonctions pour les métriques des modèles ===================

  // Fonction pour charger les métriques des modèles
  function loadModelMetrics() {
    log('INFO', 'Chargement des métriques des modèles...');
    showStatus('Chargement des métriques des modèles...', 'info');

    fetch('http://localhost:8001/api/model-metrics-summary/')
      .then(response => {
        if (!response.ok) {
          throw new Error(`Erreur HTTP: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        log('INFO', 'Métriques des modèles chargées:', data);
        displayModelMetrics(data);
      })
      .catch(error => {
        log('ERROR', 'Erreur lors du chargement des métriques:', error);
        showStatus('Erreur lors du chargement des métriques. Veuillez réessayer.', 'error');
      });
  }

  // Fonction pour afficher les métriques dans le tableau
  function displayModelMetrics(data) {
    const tableBody = document.getElementById('model-metrics-table-body');
    if (!tableBody) {
      log('ERROR', 'Élément table body non trouvé pour les métriques');
      return;
    }

    let html = '';

    // L'API renvoie un objet avec les types de modèles comme clés
    for (const type in data) {
      const metrics = data[type];

      // Formater les valeurs numériques pour l'affichage
      const mseValue = formatMetricValue(metrics.mse);
      const rmseValue = formatMetricValue(metrics.rmse);
      const r2Value = formatMetricValue(metrics.r2_score);

      html += `
        <tr>
          <td>${metrics.model_name || 'Non spécifié'}</td>
          <td>${type}</td>
          <td>Tous</td>
          <td>Toutes</td>
          <td>${mseValue}</td>
          <td>${rmseValue}</td>
          <td>${r2Value}</td>
        </tr>
      `;
    }

    if (html === '') {
      html = '<tr><td colspan="7">Aucune métrique disponible.</td></tr>';
    }

    tableBody.innerHTML = html;
    showStatus('Métriques des modèles chargées avec succès.', 'success');
  }

  // =================== Fonctions pour les prédictions ===================

 // 1. Corriger la fonction loadPredictions pour utiliser les vraies APIs
function loadPredictions() {
    const predVirusSelect = document.getElementById('pred-virus-select');
    const predLocationSelect = document.getElementById('pred-location-select');

    if (!predVirusSelect || !predLocationSelect) {
        log('ERROR', 'Sélecteurs de prédiction non trouvés');
        return;
    }

    const virusId = predVirusSelect.value;
    const locationId = predLocationSelect.value;

    if (!virusId || !locationId || virusId === 'all' || locationId === 'all') {
        const noPredictionData = document.getElementById('no-prediction-data');
        if (noPredictionData) noPredictionData.style.display = 'block';
        log('INFO', 'Aucune sélection valide pour les prédictions');
        return;
    }

    const virusName = predVirusSelect.options[predVirusSelect.selectedIndex].text;
    const locationName = predLocationSelect.options[predLocationSelect.selectedIndex].text;

    log('INFO', `Chargement des prédictions pour ${virusName} en ${locationName}...`);
    showStatus(`Chargement des prédictions pour ${virusName} en ${locationName}...`, 'info');

    // Appeler l'API forecast pour obtenir de vraies prédictions
    const requestData = {
        location_id: parseInt(locationId),
        virus_id: parseInt(virusId),
        weeks: 4
    };

    fetch('http://localhost:8001/api/predict/forecast/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Erreur HTTP: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        log('INFO', 'Prédictions reçues de l\'API:', data);
        displayPredictions(data, virusName, locationName);
    })
    .catch(error => {
        log('ERROR', 'Erreur lors du chargement des prédictions:', error);
        showStatus('Erreur lors du chargement des prédictions. Affichage de données simulées.', 'warning');
        displayMockPredictions(virusName, locationName);
    });
}
  function createNewGeoPrediction(virusId, virusName) {
    const requestData = {
        virus_id: virusId
    };

    fetch('http://localhost:8001/api/predict/geographical-spread/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Erreur HTTP: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        log('INFO', 'Nouvelle prédiction géographique créée:', data);

        // Convertir la réponse au format attendu par displayGeoPredictions
        const formattedData = {
            results: [{
                virus: virusName,
                prediction_date: data.prediction_date,
                predicted_new_locations: data.predicted_new_locations
            }]
        };

        displayGeoPredictions(formattedData, virusName);
    })
    .catch(error => {
        log('ERROR', 'Erreur lors de la création de prédictions géographiques:', error);
        showStatus('Erreur lors du chargement des prédictions géographiques. Affichage de données simulées.', 'warning');
        displayMockGeoPredictions(virusName);
    });
}

  // Fonction pour afficher des données de prédiction simulées
  function displayMockPredictions(virusName, locationName) {
    log('INFO', 'Affichage de prédictions simulées pour démonstration');

    const mockData = {
      location: locationName,
      virus: virusName,
      forecasts: [
        {
          predictions: {
            transmission_rate: virusName === 'COVID' ? 1.2 : 0.8,
            mortality_rate: virusName === 'COVID' ? 0.02 : 0.01,
            predicted_cases_next_week: virusName === 'COVID' ? 5000 : 200,
            predicted_deaths_next_week: virusName === 'COVID' ? 100 : 2
          }
        },
        {
          predictions: {
            transmission_rate: virusName === 'COVID' ? 1.1 : 0.7,
            mortality_rate: virusName === 'COVID' ? 0.018 : 0.008,
            predicted_cases_next_week: virusName === 'COVID' ? 4500 : 150,
            predicted_deaths_next_week: virusName === 'COVID' ? 81 : 1
          }
        },
        {
          predictions: {
            transmission_rate: virusName === 'COVID' ? 0.95 : 0.6,
            mortality_rate: virusName === 'COVID' ? 0.015 : 0.005,
            predicted_cases_next_week: virusName === 'COVID' ? 4000 : 100,
            predicted_deaths_next_week: virusName === 'COVID' ? 60 : 0
          }
        },
        {
          predictions: {
            transmission_rate: virusName === 'COVID' ? 0.85 : 0.5,
            mortality_rate: virusName === 'COVID' ? 0.012 : 0.003,
            predicted_cases_next_week: virusName === 'COVID' ? 3400 : 50,
            predicted_deaths_next_week: virusName === 'COVID' ? 41 : 0
          }
        }
      ]
    };

    displayPredictions(mockData, virusName, locationName);
  }

  // Fonction pour afficher les prédictions
  function displayPredictions(data, virusName, locationName) {
    const predictionList = document.getElementById('prediction-list');
    const noPredictionData = document.getElementById('no-prediction-data');

    if (!predictionList) {
      log('ERROR', 'Élément prediction-list non trouvé');
      return;
    }

    // Vérifier si des prédictions sont disponibles
    if (!data || !data.forecasts || data.forecasts.length === 0) {
      if (noPredictionData) noPredictionData.style.display = 'block';
      predictionList.innerHTML = '';
      return;
    }

    if (noPredictionData) noPredictionData.style.display = 'none';
    let html = '';

    // Parcourir les prédictions
    data.forecasts.forEach((forecast, index) => {
      const week = index + 1;
      const pred = forecast.predictions || {};

      html += `
        <div class="list-group-item">
          <div class="d-flex w-100 justify-content-between">
            <h5 class="mb-1">Semaine ${week}</h5>
          </div>
          <p class="mb-1">
            <span class="prediction-badge transmission">Rt: ${pred.transmission_rate ? pred.transmission_rate.toFixed(2) : 'N/A'}</span>
            <span class="prediction-badge mortality">Mortalité: ${pred.mortality_rate ? (pred.mortality_rate * 100).toFixed(2) + '%' : 'N/A'}</span>
          </p>
          <p class="mb-1">
            <span class="prediction-badge cases">Nouveaux cas: ${pred.predicted_cases_next_week ? pred.predicted_cases_next_week.toLocaleString() : 'N/A'}</span>
            <span class="prediction-badge deaths">Nouveaux décès: ${pred.predicted_deaths_next_week ? pred.predicted_deaths_next_week.toLocaleString() : 'N/A'}</span>
          </p>
        </div>
      `;
    });

    predictionList.innerHTML = html;
    showStatus(`Prédictions pour ${virusName} en ${locationName} chargées avec succès.`, 'success');

    // Mettre à jour le graphique de prédiction
    updatePredictionChart(data.forecasts);
  }

  // Fonction pour mettre à jour le graphique de prédiction
  function updatePredictionChart(forecasts) {
    const chartCanvas = document.getElementById('prediction-chart');
    if (!chartCanvas) {
      log('ERROR', 'Canvas prediction-chart non trouvé');
      return;
    }

    // Détruire le graphique existant s'il y en a un
    if (window.predictionChart instanceof Chart) {
      window.predictionChart.destroy();
    }

    const labels = forecasts.map((_, index) => `Semaine ${index + 1}`);
    const transmissionData = forecasts.map(f => f.predictions?.transmission_rate || 0);
    const mortalityData = forecasts.map(f => (f.predictions?.mortality_rate || 0) * 100);

    // Créer un nouveau graphique
    window.predictionChart = new Chart(chartCanvas, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Taux de transmission (Rt)',
            data: transmissionData,
            borderColor: CHART_COLORS.newCases,
            backgroundColor: 'rgba(32, 201, 151, 0.1)',
            borderWidth: 2,
            fill: true
          },
          {
            label: 'Taux de mortalité (%)',
            data: mortalityData,
            borderColor: CHART_COLORS.newDeaths,
            backgroundColor: 'rgba(255, 149, 0, 0.1)',
            borderWidth: 2,
            fill: true
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true
          }
        }
      }
    });
  }
  function loadGeoPredictions() {
    const geoPredVirusSelect = document.getElementById('geo-pred-virus-select');

    if (!geoPredVirusSelect) {
        log('ERROR', 'Sélecteur de prédiction géographique non trouvé');
        return;
    }

    const virusId = geoPredVirusSelect.value;

    if (!virusId || virusId === 'all') {
        const noGeoPredictionData = document.getElementById('no-geo-prediction-data');
        if (noGeoPredictionData) noGeoPredictionData.style.display = 'block';
        log('INFO', 'Aucune sélection valide pour les prédictions géographiques');
        return;
    }

    const virusName = geoPredVirusSelect.options[geoPredVirusSelect.selectedIndex].text;

    log('INFO', `Chargement des prédictions de propagation pour ${virusName}...`);
    showStatus(`Chargement des prédictions de propagation pour ${virusName}...`, 'info');

    // Essayer d'abord l'API geographical-predictions
    fetch(`http://localhost:8001/api/geographical-predictions/?virus=${virusName}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Erreur HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            log('INFO', 'Prédictions de propagation chargées:', data);

            if (!data || !data.results || data.results.length === 0) {
                // Si pas de données existantes, créer de nouvelles prédictions
                return createNewGeoPrediction(parseInt(virusId), virusName);
            } else {
                displayGeoPredictions(data, virusName);
            }
        })
        .catch(error => {
            log('ERROR', 'Erreur lors du chargement des prédictions géographiques:', error);
            // Essayer de créer de nouvelles prédictions
            createNewGeoPrediction(parseInt(virusId), virusName);
        });
}

  // Fonction pour simuler des données de prédiction géographique
  function displayMockGeoPredictions(virusName) {
    log('INFO', 'Affichage de prédictions géographiques simulées pour démonstration');

    const today = new Date();
    const mockData = {
      results: [
        {
          virus: virusName,
          prediction_date: new Date(today.getTime() + 7 * 24 * 60 * 60 * 1000).toISOString().substring(0, 10),
          predicted_new_locations: virusName === 'COVID' ? 3 : 1
        },
        {
          virus: virusName,
          prediction_date: new Date(today.getTime() + 14 * 24 * 60 * 60 * 1000).toISOString().substring(0, 10),
          predicted_new_locations: virusName === 'COVID' ? 5 : 2
        },
        {
          virus: virusName,
          prediction_date: new Date(today.getTime() + 21 * 24 * 60 * 60 * 1000).toISOString().substring(0, 10),
          predicted_new_locations: virusName === 'COVID' ? 2 : 1
        },
        {
          virus: virusName,
          prediction_date: new Date(today.getTime() + 28 * 24 * 60 * 60 * 1000).toISOString().substring(0, 10),
          predicted_new_locations: virusName === 'COVID' ? 1 : 0
        }
      ]
    };

    displayGeoPredictions(mockData, virusName);
  }

  // Fonction pour afficher les prédictions géographiques
  function displayGeoPredictions(data, virusName) {
    const geoPredictionList = document.getElementById('geo-prediction-list');
    const noGeoPredictionData = document.getElementById('no-geo-prediction-data');

    if (!geoPredictionList) {
      log('ERROR', 'Élément geo-prediction-list non trouvé');
      return;
    }

    // Vérifier si des prédictions sont disponibles
    if (!data || !data.results || data.results.length === 0) {
      if (noGeoPredictionData) noGeoPredictionData.style.display = 'block';
      geoPredictionList.innerHTML = '';
      return;
    }

    if (noGeoPredictionData) noGeoPredictionData.style.display = 'none';
    let html = '';

    // Parcourir les prédictions
    data.results.forEach((prediction) => {
      const date = new Date(prediction.prediction_date).toLocaleDateString('fr-FR');

      html += `
        <div class="list-group-item">
          <div class="d-flex w-100 justify-content-between">
            <h5 class="mb-1">${date}</h5>
          </div>
          <p class="mb-1">
            <span class="prediction-badge spread">Nouvelles localisations: ${prediction.predicted_new_locations}</span>
          </p>
        </div>
      `;
    });

    geoPredictionList.innerHTML = html;
    showStatus(`Prédictions de propagation pour ${virusName} chargées avec succès.`, 'success');

    // Mettre à jour le graphique de propagation
    updateGeoPredictionChart(data.results);
  }

  // Fonction pour mettre à jour le graphique de propagation
  function updateGeoPredictionChart(predictions) {
    const chartCanvas = document.getElementById('geo-prediction-chart');
    if (!chartCanvas) {
      log('ERROR', 'Canvas geo-prediction-chart non trouvé');
      return;
    }

    // Détruire le graphique existant s'il y en a un
    if (window.geoPredictionChart instanceof Chart) {
      window.geoPredictionChart.destroy();
    }

    const labels = predictions.map(p => {
      if (typeof p.prediction_date === 'string') {
        return new Date(p.prediction_date).toLocaleDateString('fr-FR');
      }
      return p.prediction_date;
    });

    const newLocationsData = predictions.map(p => p.predicted_new_locations);

    // Créer un nouveau graphique
    window.geoPredictionChart = new Chart(chartCanvas, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Nouvelles localisations prédites',
            data: newLocationsData,
            backgroundColor: CHART_COLORS.casesPerMillion,
            borderColor: CHART_COLORS.casesPerMillion,
            borderWidth: 1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            ticks: {
              precision: 0 // Afficher uniquement des nombres entiers
            }
          }
        }
      }
    });
  }

  // =================== Fonctions de test et calculs spécifiques ===================

  // Fonction pour calculer les sommes (pour le bouton "Calcul somme COVID France")
  function calculateSums(virusName, locationName) {
    log('INFO', `Calcul des sommes pour ${virusName} en ${locationName}`);

    // Récupérer les dates actuellement sélectionnées dans l'interface
    const startDateInput = document.getElementById('start-date');
    const endDateInput = document.getElementById('end-date');

    const startDate = startDateInput ? startDateInput.value : null;
    const endDate = endDateInput ? endDateInput.value : null;

    // Construire l'URL avec les mêmes paramètres que le filtrage manuel
    let url = 'http://localhost:8001/api/aggregated-data/';
    let params = new URLSearchParams();

    params.append('virus_name', virusName);
    params.append('location_name', locationName);

    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);

    url += '?' + params.toString();

    log('INFO', `URL pour le calcul des sommes: ${url}`);
    showStatus(`Calcul des sommes pour ${virusName} en ${locationName}...`, 'info');

    // Effectuer la requête
    fetch(url)
      .then(response => {
        if (!response.ok) throw new Error(`Erreur HTTP: ${response.status}`);
        return response.json();
      })
      .then(data => {
        log('INFO', 'Données pour le calcul des sommes récupérées:', data);

        // Calculer les statistiques
        const stats = calculateAggregatedStats(data);

        // Afficher les résultats
        updateCards(stats);
        updateDataTable(data);
        updateCharts(data);

        // Afficher un message de succès
        showStatus(`Calcul effectué: ${stats.newCases.toLocaleString()} nouveaux cas, ${stats.newDeaths.toLocaleString()} nouveaux décès pour ${virusName} en ${locationName}`, 'success');

        // Afficher le nombre d'enregistrements
        const recordCount = data.length;
        const container = document.querySelector('.container');
        if (container) {
          // Créer ou mettre à jour le bandeau de résultat
          let resultBanner = document.getElementById('result-banner');
          if (!resultBanner) {
            resultBanner = document.createElement('div');
            resultBanner.id = 'result-banner';
            resultBanner.className = 'alert alert-success';
            resultBanner.style.marginTop = '20px';
            container.insertBefore(resultBanner, container.firstChild);
          }

          resultBanner.textContent = `Données récupérées avec succès (${recordCount} enregistrements)`;
          resultBanner.style.display = 'block';
        }
      })
      .catch(error => {
        log('ERROR', 'Erreur lors du calcul des sommes:', error);
        showStatus(`Erreur lors du calcul: ${error.message}`, 'error');
      });
  }

// NOUVEAU: Fonction pour récupérer les données de localisation (incluant population)
function fetchLocationData(locationName) {
  log('INFO', `Récupération des données pour ${locationName}...`);

  return fetch(`http://localhost:8001/api/location-details/?name=${encodeURIComponent(locationName)}`)
    .then(response => {
      if (!response.ok) {
        throw new Error(`Erreur HTTP: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      log('INFO', `Données de localisation récupérées pour ${locationName}:`, data);
      return data;
    })
    .catch(error => {
      log('ERROR', `Erreur lors de la récupération des données de ${locationName}:`, error);
      return null;
    });
}
  // =================== Fonctions pour la validation du modèle ===================

// =================== VALIDATION DU MODÈLE - CODE COMPLET ===================

// 1. Fonction pour charger les pays disponibles
function loadAvailableCountries() {
    log('INFO', 'Chargement des pays disponibles pour la validation...');

    fetch('http://localhost:8001/api/available-countries/')
        .then(response => {
            if (!response.ok) {
                throw new Error(`Erreur HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            log('INFO', `${data.total_count} pays chargés pour la validation:`, data);
            populateCountrySelect(data.countries);
        })
        .catch(error => {
            log('ERROR', 'Erreur lors du chargement des pays:', error);
            // En cas d'erreur, utiliser la liste par défaut
            useDefaultCountries();
        });
}

// 2. Fonction pour remplir le sélecteur de pays
function populateCountrySelect(countries) {
    // 🔧 MODIFIER : Remplir TOUS les sélecteurs de validation
    const selectors = [
        'validation-location-select',           // Mortalité (existant)
        'transmission-validation-location-select' // Transmission (nouveau)
    ];

    selectors.forEach(selectorId => {
        const locationSelect = document.getElementById(selectorId);

        if (!locationSelect) {
            log('DEBUG', `Sélecteur ${selectorId} non trouvé (normal si carte pas présente)`);
            return; // Continuer avec le suivant
        }

        // Vider le sélecteur
        locationSelect.innerHTML = '<option value="">Sélectionner un pays...</option>';

        // Grouper les pays par qualité des données
        const excellentCountries = countries.filter(c => c.data_quality === 'Excellente');
        const goodCountries = countries.filter(c => c.data_quality === 'Bonne');
        const fairCountries = countries.filter(c => c.data_quality === 'Correcte');

        // Ajouter les groupes avec des optgroups
        if (excellentCountries.length > 0) {
            const excellentGroup = document.createElement('optgroup');
            excellentGroup.label = `🌟 Qualité Excellente (${excellentCountries.length} pays)`;
            excellentCountries.forEach(country => {
                const option = document.createElement('option');
                option.value = country.name;
                option.textContent = `${country.name} (${country.total_cases.toLocaleString()} cas)`;
                excellentGroup.appendChild(option);
            });
            locationSelect.appendChild(excellentGroup);
        }

        if (goodCountries.length > 0) {
            const goodGroup = document.createElement('optgroup');
            goodGroup.label = `✅ Qualité Bonne (${goodCountries.length} pays)`;
            goodCountries.forEach(country => {
                const option = document.createElement('option');
                option.value = country.name;
                option.textContent = `${country.name} (${country.total_cases.toLocaleString()} cas)`;
                goodGroup.appendChild(option);
            });
            locationSelect.appendChild(goodGroup);
        }

        if (fairCountries.length > 0) {
            const fairGroup = document.createElement('optgroup');
            fairGroup.label = `⚠️ Qualité Correcte (${fairCountries.length} pays)`;
            fairCountries.forEach(country => {
                const option = document.createElement('option');
                option.value = country.name;
                option.textContent = `${country.name} (${country.total_cases.toLocaleString()} cas)`;
                fairGroup.appendChild(option);
            });
            locationSelect.appendChild(fairGroup);
        }

        log('INFO', `Sélecteur ${selectorId} rempli avec ${countries.length} pays`);

        // Sélectionner la France par défaut après un délai
        setTimeout(() => {
            locationSelect.value = 'France';
            log('INFO', `France sélectionnée par défaut pour ${selectorId}`);
        }, 500);
    });
}

// 3. Fonction de fallback avec la liste par défaut
function useDefaultCountries() {
    log('WARN', 'Utilisation de la liste de pays par défaut');

    const defaultCountries = [
        { name: 'France', data_quality: 'Excellente' },
        { name: 'USA', data_quality: 'Excellente' },
        { name: 'UK', data_quality: 'Excellente' },
        { name: 'Germany', data_quality: 'Excellente' },
        { name: 'Italy', data_quality: 'Excellente' },
        { name: 'Spain', data_quality: 'Excellente' },
        { name: 'Brazil', data_quality: 'Excellente' },
        { name: 'India', data_quality: 'Excellente' }
    ];

    const locationSelect = document.getElementById('validation-location-select');
    if (locationSelect) {
        locationSelect.innerHTML = '<option value="">Sélectionner un pays...</option>';

        defaultCountries.forEach(country => {
            const option = document.createElement('option');
            option.value = country.name;
            option.textContent = country.name;
            locationSelect.appendChild(option);
        });

        // Sélectionner la France par défaut
        setTimeout(() => {
            locationSelect.value = 'France';
        }, 500);
    }
}

// 4. Fonction pour charger la validation du modèle
function loadModelValidation() {
    const locationSelect = document.getElementById('validation-location-select');
    const periodSelect = document.getElementById('validation-period-select');
    const statusSpan = document.getElementById('validation-status');

    if (!locationSelect || !periodSelect) {
        log('ERROR', 'Sélecteurs de validation non trouvés');
        return;
    }

    const locationName = locationSelect.value;
    const periodValue = periodSelect.value;

    if (!locationName || !periodValue) {
        showStatus('Veuillez sélectionner un pays et une période.', 'warning');
        return;
    }

    const [startDate, endDate] = periodValue.split(',');

    log('INFO', `Validation du modèle pour ${locationName} (${startDate} - ${endDate})`);
    statusSpan.textContent = 'Validation en cours...';
    statusSpan.className = 'ms-2 text-primary';

    const requestData = {
        location_name: locationName,
        start_date: startDate,
        end_date: endDate,
        max_results: 100
    };

    fetch('http://localhost:8001/api/validate-model/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Erreur HTTP: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        log('INFO', 'Résultats de validation reçus:', data);
        statusSpan.textContent = 'Validation terminée';
        statusSpan.className = 'ms-2 text-success';
        displayValidationResults(data);
    })
    .catch(error => {
        log('ERROR', 'Erreur lors de la validation:', error);
        statusSpan.textContent = 'Erreur lors de la validation';
        statusSpan.className = 'ms-2 text-danger';
        showStatus(`Erreur lors de la validation: ${error.message}`, 'error');
    });
}

// 5. Fonction pour afficher les résultats de validation
function displayValidationResults(data) {
    const resultsDiv = document.getElementById('validation-results');
    const validationList = document.getElementById('validation-list');

    if (!resultsDiv || !validationList) {
        log('ERROR', 'Éléments de résultats de validation non trouvés');
        return;
    }

    resultsDiv.style.display = 'block';
    updateValidationStats(data.statistics);
    displayValidationPredictions(data.predictions);
    updateValidationChart(data.predictions);
    showStatus(`Validation terminée: ${data.statistics.total_predictions} prédictions analysées pour ${data.location}`, 'success');
}

// 6. Fonction pour mettre à jour les statistiques de validation
function updateValidationStats(stats) {
    const elements = {
        'validation-accuracy': `${stats.median_error_relative}%`,
        'validation-count': stats.total_predictions,
        'excellent-count': stats.excellent_count,
        'good-count': stats.good_count,
        'fair-count': stats.fair_count,
        'poor-count': stats.poor_count
    };

    for (const [id, value] of Object.entries(elements)) {
        const elem = document.getElementById(id);
        if (elem) {
            elem.textContent = value;
        }
    }
}

// 7. Fonction pour afficher la liste des prédictions
function displayValidationPredictions(predictions) {
    const validationList = document.getElementById('validation-list');
    if (!validationList) return;

    let html = '';

    // Ajouter des contrôles d'affichage
    html += `
        <div class="d-flex justify-content-between align-items-center mb-3">
            <h6 class="mb-0">Détails des Prédictions (${predictions.length} total)</h6>
            <div>
                <button id="show-more-btn" class="btn btn-sm btn-outline-primary me-2">Voir plus</button>
                <button id="show-all-btn" class="btn btn-sm btn-outline-secondary">Tout voir</button>
            </div>
        </div>
    `;

    // Par défaut, afficher les 20 premières
    let displayCount = Math.min(20, predictions.length);
    let displayPredictions = predictions.slice(0, displayCount);

    displayPredictions.forEach((pred) => {
        const qualityClass = getQualityClass(pred.quality_class);
        const qualityIcon = getQualityIcon(pred.quality_class);

        html += `
            <div class="list-group-item">
                <div class="d-flex w-100 justify-content-between">
                    <h6 class="mb-1">${formatDateFr(pred.date)}</h6>
                    <small class="text-muted">Erreur: ${pred.error_relative}%</small>
                </div>
                <div class="mb-1">
                    <span class="badge ${qualityClass} me-1">${qualityIcon} ${pred.quality}</span>
                    <span class="prediction-badge mortality">Prédit: ${pred.predicted_mortality}%</span>
                    <span class="prediction-badge transmission">Réel: ${pred.actual_mortality}%</span>
                </div>
                <small class="text-muted">
                    Cas: ${pred.new_cases_7d.toLocaleString()} (7j) | 
                    Décès: ${pred.new_deaths_7d.toLocaleString()} (7j) |
                    Total: ${pred.total_cases.toLocaleString()} cas
                </small>
            </div>
        `;
    });

    if (predictions.length > displayCount) {
        html += `
            <div class="list-group-item text-center text-muted">
                <small>... et ${predictions.length - displayCount} autres prédictions</small>
            </div>
        `;
    }

    validationList.innerHTML = html;

    // Ajouter les événements pour "Voir plus" et "Tout voir"
    const showMoreBtn = document.getElementById('show-more-btn');
    const showAllBtn = document.getElementById('show-all-btn');

    if (showMoreBtn) {
        showMoreBtn.addEventListener('click', () => showMorePredictions(predictions, 50));
    }

    if (showAllBtn) {
        showAllBtn.addEventListener('click', () => showAllPredictions(predictions));
    }
}

// 8. Fonction pour afficher plus de prédictions
function showMorePredictions(predictions, count) {
    const validationList = document.getElementById('validation-list');
    if (!validationList) return;

    let html = `
        <div class="d-flex justify-content-between align-items-center mb-3">
            <h6 class="mb-0">Détails des Prédictions (${predictions.length} total)</h6>
            <div>
                <button id="show-all-btn" class="btn btn-sm btn-outline-secondary me-2">Tout voir</button>
                <button id="show-less-btn" class="btn btn-sm btn-outline-danger">Réduire</button>
            </div>
        </div>
    `;

    const displayCount = Math.min(count, predictions.length);
    const displayPredictions = predictions.slice(0, displayCount);

    displayPredictions.forEach((pred) => {
        const qualityClass = getQualityClass(pred.quality_class);
        const qualityIcon = getQualityIcon(pred.quality_class);

        html += `
            <div class="list-group-item">
                <div class="d-flex w-100 justify-content-between">
                    <h6 class="mb-1">${formatDateFr(pred.date)}</h6>
                    <small class="text-muted">Erreur: ${pred.error_relative}%</small>
                </div>
                <div class="mb-1">
                    <span class="badge ${qualityClass} me-1">${qualityIcon} ${pred.quality}</span>
                    <span class="prediction-badge mortality">Prédit: ${pred.predicted_mortality}%</span>
                    <span class="prediction-badge transmission">Réel: ${pred.actual_mortality}%</span>
                </div>
                <small class="text-muted">
                    Cas: ${pred.new_cases_7d.toLocaleString()} (7j) | 
                    Décès: ${pred.new_deaths_7d.toLocaleString()} (7j) |
                    Total: ${pred.total_cases.toLocaleString()} cas
                </small>
            </div>
        `;
    });

    if (predictions.length > displayCount) {
        html += `
            <div class="list-group-item text-center text-muted">
                <small>... et ${predictions.length - displayCount} autres prédictions</small>
            </div>
        `;
    }

    validationList.innerHTML = html;

    const showAllBtn = document.getElementById('show-all-btn');
    const showLessBtn = document.getElementById('show-less-btn');

    if (showAllBtn) {
        showAllBtn.addEventListener('click', () => showAllPredictions(predictions));
    }

    if (showLessBtn) {
        showLessBtn.addEventListener('click', () => displayValidationPredictions(predictions));
    }
}

// 9. Fonction pour afficher toutes les prédictions
function showAllPredictions(predictions) {
    const validationList = document.getElementById('validation-list');
    if (!validationList) return;

    let html = `
        <div class="d-flex justify-content-between align-items-center mb-3">
            <h6 class="mb-0">Toutes les Prédictions (${predictions.length})</h6>
            <div>
                <button id="show-less-btn" class="btn btn-sm btn-outline-danger me-2">Réduire</button>
                <button id="export-csv-btn" class="btn btn-sm btn-outline-success">Exporter CSV</button>
            </div>
        </div>
    `;

    predictions.forEach((pred) => {
        const qualityClass = getQualityClass(pred.quality_class);
        const qualityIcon = getQualityIcon(pred.quality_class);

        html += `
            <div class="list-group-item">
                <div class="d-flex w-100 justify-content-between">
                    <h6 class="mb-1">${formatDateFr(pred.date)}</h6>
                    <small class="text-muted">Erreur: ${pred.error_relative}%</small>
                </div>
                <div class="mb-1">
                    <span class="badge ${qualityClass} me-1">${qualityIcon} ${pred.quality}</span>
                    <span class="prediction-badge mortality">Prédit: ${pred.predicted_mortality}%</span>
                    <span class="prediction-badge transmission">Réel: ${pred.actual_mortality}%</span>
                </div>
                <small class="text-muted">
                    Cas: ${pred.new_cases_7d.toLocaleString()} (7j) | 
                    Décès: ${pred.new_deaths_7d.toLocaleString()} (7j) |
                    Total: ${pred.total_cases.toLocaleString()} cas
                </small>
            </div>
        `;
    });

    validationList.innerHTML = html;

    const showLessBtn = document.getElementById('show-less-btn');
    const exportCsvBtn = document.getElementById('export-csv-btn');

    if (showLessBtn) {
        showLessBtn.addEventListener('click', () => displayValidationPredictions(predictions));
    }

    if (exportCsvBtn) {
        exportCsvBtn.addEventListener('click', () => exportPredictionsToCSV(predictions));
    }
}

// 10. Fonction pour exporter en CSV
function exportPredictionsToCSV(predictions) {
    const headers = ['Date', 'Taux_Predit_%', 'Taux_Reel_%', 'Erreur_Relative_%', 'Qualite', 'Cas_7j', 'Deces_7j', 'Total_Cas'];

    let csvContent = headers.join(',') + '\n';

    predictions.forEach(pred => {
        const row = [
            pred.date,
            pred.predicted_mortality,
            pred.actual_mortality,
            pred.error_relative,
            pred.quality,
            pred.new_cases_7d,
            pred.new_deaths_7d,
            pred.total_cases
        ];
        csvContent += row.join(',') + '\n';
    });

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `validation_predictions_${new Date().toISOString().slice(0,10)}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// 11. Fonction pour obtenir la classe CSS selon la qualité
function getQualityClass(qualityClass) {
    const classes = {
        'excellent': 'bg-success text-white',
        'good': 'bg-info text-white',
        'fair': 'bg-warning text-dark',
        'poor': 'bg-danger text-white'
    };
    return classes[qualityClass] || 'bg-secondary text-white';
}

// 12. Fonction pour obtenir l'icône selon la qualité
function getQualityIcon(qualityClass) {
    const icons = {
        'excellent': '🎯',
        'good': '✅',
        'fair': '⚠️',
        'poor': '❌'
    };
    return icons[qualityClass] || '📊';
}

// 13. Fonction pour mettre à jour le graphique de validation
function updateValidationChart(predictions) {
    const chartCanvas = document.getElementById('validation-chart');
    if (!chartCanvas) {
        log('ERROR', 'Canvas validation-chart non trouvé');
        return;
    }

    if (window.validationChart instanceof Chart) {
        window.validationChart.destroy();
    }

    const displayData = predictions.slice(-30);
    const labels = displayData.map(p => formatShortDate(p.date));
    const predictedData = displayData.map(p => p.predicted_mortality);
    const actualData = displayData.map(p => p.actual_mortality);
    const errorData = displayData.map(p => p.error_relative);

    window.validationChart = new Chart(chartCanvas, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Taux Prédit (%)',
                    data: predictedData,
                    borderColor: '#0077b6',
                    backgroundColor: 'rgba(0, 119, 182, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    yAxisID: 'y'
                },
                {
                    label: 'Taux Réel (%)',
                    data: actualData,
                    borderColor: '#d00000',
                    backgroundColor: 'rgba(208, 0, 0, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    yAxisID: 'y'
                },
                {
                    label: 'Erreur Relative (%)',
                    data: errorData,
                    borderColor: '#ff9500',
                    backgroundColor: 'rgba(255, 149, 0, 0.1)',
                    borderWidth: 1,
                    fill: true,
                    yAxisID: 'y1',
                    type: 'bar'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        afterBody: function(context) {
                            const index = context[0].dataIndex;
                            const pred = displayData[index];
                            return [
                                `Qualité: ${pred.quality}`,
                                `Cas (7j): ${pred.new_cases_7d.toLocaleString()}`,
                                `Décès (7j): ${pred.new_deaths_7d.toLocaleString()}`
                            ];
                        }
                    }
                },
                title: {
                    display: true,
                    text: 'Validation du Modèle: Prédictions vs Réalité'
                }
            },
            scales: {
                x: {
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Taux de Mortalité (%)'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Erreur Relative (%)'
                    },
                    grid: {
                        drawOnChartArea: false,
                    },
                }
            }
        }
    });
}

// 14. Fonction d'initialisation pour la validation
function initializeValidation() {
    const validateBtn = document.getElementById('validate-model-btn');

    if (validateBtn) {
        validateBtn.addEventListener('click', loadModelValidation);
        log('INFO', 'Écouteur de validation ajouté');
    } else {
        log('ERROR', 'Bouton validate-model-btn non trouvé');
    }

    // Charger tous les pays disponibles pour la validation
    loadAvailableCountries();
}

function loadTransmissionModelValidation() {
    const locationSelect = document.getElementById('transmission-validation-location-select');
    const periodSelect = document.getElementById('transmission-validation-period-select');
    const statusSpan = document.getElementById('transmission-validation-status');

    if (!locationSelect || !periodSelect) {
        log('ERROR', 'Sélecteurs de validation transmission non trouvés');
        return;
    }

    const locationName = locationSelect.value;
    const periodValue = periodSelect.value;

    if (!locationName || !periodValue) {
        showStatus('Veuillez sélectionner un pays et une période pour la validation transmission.', 'warning');
        return;
    }

    const [startDate, endDate] = periodValue.split(',');

    log('INFO', `Validation du modèle de transmission pour ${locationName} (${startDate} - ${endDate})`);
    statusSpan.textContent = 'Validation transmission en cours...';
    statusSpan.className = 'ms-2 text-primary';

    const requestData = {
        location_name: locationName,
        start_date: startDate,
        end_date: endDate,
        max_results: 100
    };

    fetch('http://localhost:8001/api/validate-transmission-model/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Erreur HTTP: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        log('INFO', 'Résultats de validation transmission reçus:', data);
        statusSpan.textContent = 'Validation transmission terminée';
        statusSpan.className = 'ms-2 text-success';
        displayTransmissionValidationResults(data);
    })
    .catch(error => {
        log('ERROR', 'Erreur lors de la validation transmission:', error);
        statusSpan.textContent = 'Erreur lors de la validation transmission';
        statusSpan.className = 'ms-2 text-danger';
        showStatus(`Erreur lors de la validation transmission: ${error.message}`, 'error');
    });
}

// 2. Fonction pour afficher les résultats de validation transmission
function displayTransmissionValidationResults(data) {
    const resultsDiv = document.getElementById('transmission-validation-results');
    const validationList = document.getElementById('transmission-validation-list');

    if (!resultsDiv || !validationList) {
        log('ERROR', 'Éléments de résultats de validation transmission non trouvés');
        return;
    }

    resultsDiv.style.display = 'block';
    updateTransmissionValidationStats(data.statistics);
    displayTransmissionValidationPredictions(data.predictions);
    updateTransmissionValidationChart(data.predictions);
    showStatus(`Validation transmission terminée: ${data.statistics.total_predictions} prédictions Rt analysées pour ${data.location}`, 'success');
}

// 3. Fonction pour mettre à jour les statistiques de validation transmission
function updateTransmissionValidationStats(stats) {
    const elements = {
        'transmission-validation-accuracy': `${stats.median_error_relative}%`,
        'transmission-validation-count': stats.total_predictions,
        'transmission-excellent-count': stats.excellent_count,
        'transmission-good-count': stats.good_count,
        'transmission-fair-count': stats.fair_count,
        'transmission-poor-count': stats.poor_count
    };

    for (const [id, value] of Object.entries(elements)) {
        const elem = document.getElementById(id);
        if (elem) {
            elem.textContent = value;
        }
    }
}

// 4. Fonction pour afficher la liste des prédictions transmission
function displayTransmissionValidationPredictions(predictions) {
    const validationList = document.getElementById('transmission-validation-list');
    if (!validationList) return;

    let html = '';

    // Ajouter des contrôles d'affichage
    html += `
        <div class="d-flex justify-content-between align-items-center mb-3">
            <h6 class="mb-0">Détails des Prédictions Rt (${predictions.length} total)</h6>
            <div>
                <button id="transmission-show-more-btn" class="btn btn-sm btn-outline-primary me-2">Voir plus</button>
                <button id="transmission-show-all-btn" class="btn btn-sm btn-outline-secondary">Tout voir</button>
            </div>
        </div>
    `;

    // Par défaut, afficher les 20 premières
    let displayCount = Math.min(20, predictions.length);
    let displayPredictions = predictions.slice(0, displayCount);

    displayPredictions.forEach((pred) => {
        const qualityClass = getQualityClass(pred.quality_class);
        const qualityIcon = getQualityIcon(pred.quality_class);

        // Spécifique transmission: afficher Rt prédit vs Rt futur réel
        html += `
            <div class="list-group-item">
                <div class="d-flex w-100 justify-content-between">
                    <h6 class="mb-1">${formatDateFr(pred.date)}</h6>
                    <small class="text-muted">Erreur: ${pred.error_relative}%</small>
                </div>
                <div class="mb-1">
                    <span class="badge ${qualityClass} me-1">${qualityIcon} ${pred.quality}</span>
                    <span class="prediction-badge transmission">Rt Prédit: ${pred.predicted_rt}</span>
                    <span class="prediction-badge cases">Rt Futur Réel: ${pred.actual_rt_future}</span>
                </div>
                <small class="text-muted">
                    État Futur: ${pred.future_epidemic_state} | 
                    Cas: ${pred.new_cases_7d.toLocaleString()} (7j) |
                    Total: ${pred.total_cases.toLocaleString()} cas
                </small>
            </div>
        `;
    });

    if (predictions.length > displayCount) {
        html += `
            <div class="list-group-item text-center text-muted">
                <small>... et ${predictions.length - displayCount} autres prédictions Rt</small>
            </div>
        `;
    }

    validationList.innerHTML = html;

    // Ajouter les événements pour "Voir plus" et "Tout voir"
    const showMoreBtn = document.getElementById('transmission-show-more-btn');
    const showAllBtn = document.getElementById('transmission-show-all-btn');

    if (showMoreBtn) {
        showMoreBtn.addEventListener('click', () => showMoreTransmissionPredictions(predictions, 50));
    }

    if (showAllBtn) {
        showAllBtn.addEventListener('click', () => showAllTransmissionPredictions(predictions));
    }
}

// 5. Fonction pour afficher plus de prédictions transmission
function showMoreTransmissionPredictions(predictions, count) {
    const validationList = document.getElementById('transmission-validation-list');
    if (!validationList) return;

    let html = `
        <div class="d-flex justify-content-between align-items-center mb-3">
            <h6 class="mb-0">Détails des Prédictions Rt (${predictions.length} total)</h6>
            <div>
                <button id="transmission-show-all-btn" class="btn btn-sm btn-outline-secondary me-2">Tout voir</button>
                <button id="transmission-show-less-btn" class="btn btn-sm btn-outline-danger">Réduire</button>
            </div>
        </div>
    `;

    const displayCount = Math.min(count, predictions.length);
    const displayPredictions = predictions.slice(0, displayCount);

    displayPredictions.forEach((pred) => {
        const qualityClass = getQualityClass(pred.quality_class);
        const qualityIcon = getQualityIcon(pred.quality_class);

        html += `
            <div class="list-group-item">
                <div class="d-flex w-100 justify-content-between">
                    <h6 class="mb-1">${formatDateFr(pred.date)}</h6>
                    <small class="text-muted">Erreur: ${pred.error_relative}%</small>
                </div>
                <div class="mb-1">
                    <span class="badge ${qualityClass} me-1">${qualityIcon} ${pred.quality}</span>
                    <span class="prediction-badge transmission">Rt Prédit: ${pred.predicted_rt}</span>
                    <span class="prediction-badge cases">Rt Futur Réel: ${pred.actual_rt_future}</span>
                </div>
                <small class="text-muted">
                    État Futur: ${pred.future_epidemic_state} | 
                    Cas: ${pred.new_cases_7d.toLocaleString()} (7j) |
                    Total: ${pred.total_cases.toLocaleString()} cas
                </small>
            </div>
        `;
    });

    if (predictions.length > displayCount) {
        html += `
            <div class="list-group-item text-center text-muted">
                <small>... et ${predictions.length - displayCount} autres prédictions Rt</small>
            </div>
        `;
    }

    validationList.innerHTML = html;

    const showAllBtn = document.getElementById('transmission-show-all-btn');
    const showLessBtn = document.getElementById('transmission-show-less-btn');

    if (showAllBtn) {
        showAllBtn.addEventListener('click', () => showAllTransmissionPredictions(predictions));
    }

    if (showLessBtn) {
        showLessBtn.addEventListener('click', () => displayTransmissionValidationPredictions(predictions));
    }
}

// 6. Fonction pour afficher toutes les prédictions transmission
function showAllTransmissionPredictions(predictions) {
    const validationList = document.getElementById('transmission-validation-list');
    if (!validationList) return;

    let html = `
        <div class="d-flex justify-content-between align-items-center mb-3">
            <h6 class="mb-0">Toutes les Prédictions Rt (${predictions.length})</h6>
            <div>
                <button id="transmission-show-less-btn" class="btn btn-sm btn-outline-danger me-2">Réduire</button>
                <button id="transmission-export-csv-btn" class="btn btn-sm btn-outline-success">Exporter CSV</button>
            </div>
        </div>
    `;

    predictions.forEach((pred) => {
        const qualityClass = getQualityClass(pred.quality_class);
        const qualityIcon = getQualityIcon(pred.quality_class);

        html += `
            <div class="list-group-item">
                <div class="d-flex w-100 justify-content-between">
                    <h6 class="mb-1">${formatDateFr(pred.date)}</h6>
                    <small class="text-muted">Erreur: ${pred.error_relative}%</small>
                </div>
                <div class="mb-1">
                    <span class="badge ${qualityClass} me-1">${qualityIcon} ${pred.quality}</span>
                    <span class="prediction-badge transmission">Rt Prédit: ${pred.predicted_rt}</span>
                    <span class="prediction-badge cases">Rt Futur Réel: ${pred.actual_rt_future}</span>
                </div>
                <small class="text-muted">
                    État Futur: ${pred.future_epidemic_state} | 
                    Cas: ${pred.new_cases_7d.toLocaleString()} (7j) |
                    Total: ${pred.total_cases.toLocaleString()} cas
                </small>
            </div>
        `;
    });

    validationList.innerHTML = html;

    const showLessBtn = document.getElementById('transmission-show-less-btn');
    const exportCsvBtn = document.getElementById('transmission-export-csv-btn');

    if (showLessBtn) {
        showLessBtn.addEventListener('click', () => displayTransmissionValidationPredictions(predictions));
    }

    if (exportCsvBtn) {
        exportCsvBtn.addEventListener('click', () => exportTransmissionPredictionsToCSV(predictions));
    }
}

// 7. Fonction pour exporter les prédictions transmission en CSV
function exportTransmissionPredictionsToCSV(predictions) {
    const headers = ['Date', 'Rt_Predit', 'Rt_Futur_Reel', 'Erreur_Relative_%', 'Qualite', 'Etat_Futur', 'Cas_7j', 'Total_Cas'];

    let csvContent = headers.join(',') + '\n';

    predictions.forEach(pred => {
        const row = [
            pred.date,
            pred.predicted_rt,
            pred.actual_rt_future,
            pred.error_relative,
            pred.quality,
            pred.future_epidemic_state,
            pred.new_cases_7d,
            pred.total_cases
        ];
        csvContent += row.join(',') + '\n';
    });

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `validation_transmission_predictions_${new Date().toISOString().slice(0,10)}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// 8. Fonction pour mettre à jour le graphique de validation transmission
function updateTransmissionValidationChart(predictions) {
    const chartCanvas = document.getElementById('transmission-validation-chart');
    if (!chartCanvas) {
        log('ERROR', 'Canvas transmission-validation-chart non trouvé');
        return;
    }

    if (window.transmissionValidationChart instanceof Chart) {
        window.transmissionValidationChart.destroy();
    }

    const displayData = predictions.slice(-30);
    const labels = displayData.map(p => formatShortDate(p.date));
    const predictedData = displayData.map(p => p.predicted_rt);
    const actualData = displayData.map(p => p.actual_rt_future);
    const errorData = displayData.map(p => p.error_relative);

    window.transmissionValidationChart = new Chart(chartCanvas, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Rt Prédit',
                    data: predictedData,
                    borderColor: '#0077b6',
                    backgroundColor: 'rgba(0, 119, 182, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    yAxisID: 'y'
                },
                {
                    label: 'Rt Futur Réel',
                    data: actualData,
                    borderColor: '#d00000',
                    backgroundColor: 'rgba(208, 0, 0, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    yAxisID: 'y'
                },
                {
                    label: 'Erreur Relative (%)',
                    data: errorData,
                    borderColor: '#ff9500',
                    backgroundColor: 'rgba(255, 149, 0, 0.1)',
                    borderWidth: 1,
                    fill: true,
                    yAxisID: 'y1',
                    type: 'bar'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        afterBody: function(context) {
                            const index = context[0].dataIndex;
                            const pred = displayData[index];
                            return [
                                `Qualité: ${pred.quality}`,
                                `État Futur: ${pred.future_epidemic_state}`,
                                `Cas (7j): ${pred.new_cases_7d.toLocaleString()}`
                            ];
                        }
                    }
                },
                title: {
                    display: true,
                    text: 'Validation Modèle Transmission: Rt Prédit vs Rt Futur Réel'
                }
            },
            scales: {
                x: {
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Taux de Reproduction (Rt)'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Erreur Relative (%)'
                    },
                    grid: {
                        drawOnChartArea: false,
                    },
                }
            }
        }
    });
}

// 9. Fonction d'initialisation pour la validation transmission
function initializeTransmissionValidation() {
    const validateTransmissionBtn = document.getElementById('validate-transmission-model-btn');

    if (validateTransmissionBtn) {
        validateTransmissionBtn.addEventListener('click', loadTransmissionModelValidation);
        log('INFO', 'Écouteur de validation transmission ajouté');
    } else {
        log('ERROR', 'Bouton validate-transmission-model-btn non trouvé');
    }

    // Charger les pays disponibles (réutiliser la fonction existante)
    loadAvailableCountries();
}





function initialize() {
    console.log("Tentative d'appel à loadModelMetrics()...");
    try {
        loadModelMetrics();
    } catch (error) {
        console.error("Erreur lors du chargement des métriques:", error);
    }

    loadViruses();
    loadAllLocations();

    // Mettre à jour le sélecteur de dates au changement de virus
    if (virusSelect) {
        virusSelect.addEventListener('change', updateDateSelector);
    }

    updateDateSelector();

    // Ajouter les écouteurs d'événements pour les prédictions
    const predVirusSelect = document.getElementById('pred-virus-select');
    const predLocationSelect = document.getElementById('pred-location-select');
    const geoPredVirusSelect = document.getElementById('geo-pred-virus-select');

    if (predVirusSelect && predLocationSelect) {
        predVirusSelect.addEventListener('change', loadPredictions);
        predLocationSelect.addEventListener('change', loadPredictions);
    }

    if (geoPredVirusSelect) {
        geoPredVirusSelect.addEventListener('change', loadGeoPredictions);
    }

    // 🆕 NOUVEAU: Initialiser la validation du modèle
    initializeValidation();
    initializeTransmissionValidation();

    log('INFO', 'Initialisation terminée');
}

// Démarrer l'application - UNE SEULE FOIS à la fin
initialize();

});