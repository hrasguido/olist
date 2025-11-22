# src/con.py
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# CARGAR .env DESDE /workspace
load_dotenv("/workspace/.env")  # ¡ESTA LÍNEA ES CLAVE!

# Verificar que se cargaron
print("USER:", os.getenv('POSTGRES_USER'))
print("PASSWORD:", os.getenv('POSTGRES_PASSWORD'))
print("DB:", os.getenv('POSTGRES_DB'))

# Conexión
engine = create_engine(
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db:5432/{os.getenv('POSTGRES_DB')}"
)

try:
    with engine.connect() as conn:
        version = conn.execute(text("SELECT version();")).fetchone()[0]
        print("CONEXIÓN EXITOSA →", version)
except Exception as e:
    print("Error al conectar:", e)