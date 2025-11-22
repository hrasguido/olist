-- ./init/00_create_medallion.sql
-- Se ejecuta automáticamente al levantar PostgreSQL

-- Crear bases de datos del anillo
CREATE DATABASE bronze;
CREATE DATABASE silver;
CREATE DATABASE gold;

-- Schemas dentro de cada base
\c bronze
CREATE SCHEMA IF NOT EXISTS raw;

\c silver
CREATE SCHEMA IF NOT EXISTS curated;

\c gold
CREATE SCHEMA IF NOT EXISTS dm;