-- PostgreSQL Script for schema "pandemics"

-- Création du schéma "pandemics" s'il n'existe pas déjà
CREATE SCHEMA IF NOT EXISTS pandemics;

-- Création de la table "location"
CREATE TABLE IF NOT EXISTS pandemics.location (
  id SERIAL PRIMARY KEY,       -- "SERIAL" remplace AUTO_INCREMENT et crée automatiquement un entier séquentiel
  name VARCHAR(45) NOT NULL,
  iso_code VARCHAR(45),
  population BIGINT            -- NOUVEAU: Population du pays
);

-- Création de la table "virus"
-- Ici, on suppose que l'identifiant sera fourni manuellement
CREATE TABLE IF NOT EXISTS pandemics.virus (
  id INT PRIMARY KEY,
  name VARCHAR(45) NOT NULL
);

-- Création de la table "worldmeter"
CREATE TABLE IF NOT EXISTS pandemics.worldmeter (
  id SERIAL,  -- Auto-incrément pour worldmeter
  date DATE,
  total_cases BIGINT,
  total_deaths BIGINT,
  new_cases BIGINT,
  new_deaths BIGINT,
  new_cases_smoothed NUMERIC,
  new_deaths_smoothed NUMERIC,
  new_cases_per_million NUMERIC,
  total_cases_per_million NUMERIC,
  new_cases_smoothed_per_million NUMERIC,
  new_deaths_per_million NUMERIC,
  total_deaths_per_million NUMERIC,
  new_deaths_smoothed_per_million NUMERIC,
  location_id INT NOT NULL,
  virus_id INT NOT NULL,
  PRIMARY KEY (id),
  CONSTRAINT unique_date_location_virus UNIQUE (date, location_id, virus_id),
  CONSTRAINT fk_worldmeter_location
    FOREIGN KEY (location_id)
    REFERENCES pandemics.location (id)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT fk_worldmeter_virus
    FOREIGN KEY (virus_id)
    REFERENCES pandemics.virus (id)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION
);