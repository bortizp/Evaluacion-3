import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import sqlite3  # Importamos la librería de SQLite
from dataLoaderCleaner import clean_data
# --- Tu función clean_data() va aquí (asumo que se llama y retorna el df limpio) ---
# ... (el código de clean_data() que proporcionaste) ...

# ----------------------------------------------------------------------------------
# NUEVA FUNCIÓN: Almacenar los datos limpios en SQLite
# ----------------------------------------------------------------------------------
def save_to_sqlite(df: pd.DataFrame, db_name: str = 'proyecto_nn.db', table_name: str = 'datos_limpios'):
    """
    Guarda un DataFrame de Pandas en una tabla de una base de datos SQLite.
    """
    try:
        # Conexión: el archivo se crea si no existe
        conn = sqlite3.connect(db_name)
        
        # Exportar el DataFrame a la tabla SQL
        # if_exists='replace' es ideal para el primer uso o para refrescar datos
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        print(f"\n✅ Datos exportados con éxito a SQLite en el archivo '{db_name}'.")
        print(f"Tabla creada/reemplazada: '{table_name}'. Total: {len(df)} registros.")
        
        conn.close()
    except Exception as e:
        print(f"Error al guardar en SQLite: {e}")

# ----------------------------------------------------------------------------------
# Función para cargar datos desde SQLite
def load_data_from_sql(db_name: str = 'proyecto_nn.db', table_name: str = 'datos_limpios'):
    """
    Realiza una consulta SQL selectiva y carga los resultados en un DataFrame.
    """
    try:
        conn = sqlite3.connect(db_name)
        
        # Consulta SQL para seleccionar solo las variables numéricas clave
        sql_query = f"""
        SELECT 
            clientes_facturados, 
            e1_kwh, 
            e2_kwh, 
            energia_kwh 
        FROM {table_name}
        -- Opcional: Filtramos los datos de una región específica si el modelo es regional
        -- WHERE region = 'Región Metropolitana de Santiago' 
        -- Limitamos a 500 registros para una demostración rápida (quitamos esto para el entrenamiento real)
        LIMIT 500000; 
        """
        
        # Pandas lee el resultado de la consulta SQL directamente
        df_modelado = pd.read_sql_query(sql_query, conn)
        
        conn.close()
        
        print(f"\n📥 Datos cargados de SQL: {len(df_modelado)} registros para el modelado.")
        return df_modelado
        
    except Exception as e:
        print(f"Error al cargar datos desde SQL: {e}")
        return None
# ----------------------------------------------------------------------------------
# Función para consultar y graficar el top consumo por región
def top_consumo_por_region(db_name: str = 'proyecto_nn.db', table_name: str = 'datos_limpios'):
    """
    Consulta la base de datos SQLite para obtener el consumo total 
    agregado por región y lo grafica.
    """
    try:
        conn = sqlite3.connect(db_name)
        
        # Consulta SQL: Suma el consumo (energia_kwh) agrupado por región
        sql_query = f"""
        SELECT 
            region, 
            SUM(energia_kwh) AS Consumo_Total
        FROM {table_name}
        GROUP BY 
            region
        ORDER BY 
            Consumo_Total DESC
        LIMIT 10;
        """
        
        # Pandas carga el resultado de la consulta SQL
        df_top_consumo = pd.read_sql_query(sql_query, conn)
        
        conn.close()
        
        return df_top_consumo
        
    except Exception as e:
        print(f"Error al ejecutar la consulta SQL o conectar: {e}")
        return pd.DataFrame() # Retorna un DataFrame vacío en caso de error
# ----------------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL DE EJECUCIÓN
# ----------------------------------------------------------------------------------
if __name__ == '__main__':
    # 1. Limpiar los datos
    df_limpio = clean_data()  # Ejecutar la limpieza

    # 2. Guardar el DataFrame limpio en SQLite
    if df_limpio is not None and not df_limpio.empty:
        save_to_sqlite(df_limpio)
    # Ejecución de la demostración:
    df_para_modelo = load_data_from_sql()
    if df_para_modelo is not None:
        print(df_para_modelo.head())
    df_resultado_sql = top_consumo_por_region()
    if not df_resultado_sql.empty:
        print("\nResultados de la consulta SQL (Top 10 Consumo por Región):")
        print(df_resultado_sql)
        
        # Prepara los datos para el gráfico
        regiones = df_resultado_sql['region']
        consumo = df_resultado_sql['Consumo_Total']
        
        # Creación del Gráfico de Barras
        plt.figure(figsize=(12, 6))
        plt.barh(regiones, consumo, color='skyblue')
        plt.xlabel('Consumo Total de Energía (kWh)', fontsize=12)
        plt.title('Top 10 Regiones con Mayor Consumo Total Agregado', fontsize=14)
        plt.gca().invert_yaxis() # Muestra la región con mayor consumo arriba
        plt.tight_layout()
        plt.show()
# ----------------------------------------------------------------------------------