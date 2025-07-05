#!/bin/bash

# Variables d'environnement
DB_NAME=${POSTGRES_DB:-pandemies}
DB_USER=${POSTGRES_USER:-user}
DB_PASSWORD=${POSTGRES_PASSWORD:-guigui}
DB_HOST=${POSTGRES_HOST:-postgres}
DB_PORT=${POSTGRES_PORT:-5432}
BACKUP_FILE=$1

# Vérifier si un fichier de sauvegarde a été fourni
if [ -z "$BACKUP_FILE" ]; then
  echo "Usage: $0 <backup_file>"
  exit 1
fi

# Restaurer la base de données
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME < $BACKUP_FILE

if [ $? -eq 0 ]; then
  echo "Restauration de la base de données réussie à partir de : $BACKUP_FILE"
else
  echo "Erreur lors de la restauration de la base de données."
  exit 1
fi


