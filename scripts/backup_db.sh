#!/bin/bash

# Variables d'environnement
DB_NAME=${POSTGRES_DB:-pandemies}
DB_USER=${POSTGRES_USER:-user}
DB_PASSWORD=${POSTGRES_PASSWORD:-guigui}
DB_HOST=${POSTGRES_HOST:-postgres}
DB_PORT=${POSTGRES_PORT:-5432}
BACKUP_DIR="/opt/backups"
DATE=$(date +%Y%m%d%H%M%S)
BACKUP_FILE="$BACKUP_DIR/$DB_NAME-$DATE.sql"

# Créer le répertoire de sauvegarde si il n'existe pas
mkdir -p $BACKUP_DIR

# Exporter la base de données
PGPASSWORD=$DB_PASSWORD pg_dump -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME > $BACKUP_FILE

if [ $? -eq 0 ]; then
  echo "Sauvegarde de la base de données réussie : $BACKUP_FILE"
else
  echo "Erreur lors de la sauvegarde de la base de données."
  exit 1
fi

# Supprimer les anciennes sauvegardes (garder les 7 dernières)
find $BACKUP_DIR -name "$DB_NAME-*.sql" -mtime +7 -delete

echo "Nettoyage des anciennes sauvegardes terminé."


