import pandas as pd
import numpy as np
import os
import sys

def clean_data():
    """
    Limpia el DataFrame proporcionado realizando las siguientes operaciones:
    1. Estandariza los nombres de las columnas a minúsculas.
    2. Elimina filas duplicadas.
    3. Maneja valores faltantes:
         - Para columnas numéricas, rellena con la media de la columna.
         - Para columnas categóricas, rellena con la moda de la columna.
    entra:
        df (pd.DataFrame): El DataFrame a limpiar.
    retorna:
        pd.DataFrame: El DataFrame limpio.    
   """

    try:
        scriptDir = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(scriptDir, "DatosFacturas.csv")  #Asignando direccion del archivo
        csv = pd.read_csv(file, sep=';', encoding='latin-1') #leyendo archivo con pd y su separador e encoding
        df = pd.DataFrame(csv) 
        df.rename(columns={'ï»¿anio': 'anio'}, inplace=True) #Renombrando columna mala
        for col in df.select_dtypes(include=['object']).columns:
            # Esto intenta revertir la doble codificacion mala
            df[col] = df[col].astype(str).str.encode('latin-1').str.decode('utf-8', errors='ignore')
            # Limpiar espacios en blanco al inicio/final
            df[col] = df[col].str.strip()
            # Correccion manual de nombres de regiones mal codificados
            region_mapping = {
                'RegiÃ³n del Libertador Gral. Bernardo Oâ€™Higgins': 'Región del Libertador Gral. Bernardo O\'Higgins',
                'RegiÃ³n del BiobÃ\xado': 'Región del Biobío',
                'RegiÃ³n del Maule': 'Región del Maule',
                'RegiÃ³n de ValparaÃ\xadso': 'Región de Valparaíso',
                'RegiÃ³n Metropolitana de Santiago': 'Región Metropolitana de Santiago',
                'RegiÃ³n de La AraucanÃ\xada': 'Región de La Araucanía',
                'RegiÃ³n de Los Lagos': 'Región de Los Lagos',
                'RegiÃ³n de Atacama': 'Región de Atacama',
                'RegiÃ³n AisÃ©n del Gral.Carlos IbÃ¡Ã±ez del Campo': 'Región Aisén del Gral. Carlos Ibáñez del Campo',
                'RegiÃ³n de Coquimbo': 'Región de Coquimbo',
                'RegiÃ³n de Antofagasta': 'Región de Antofagasta',
                'RegiÃ³n de Arica y Parinacota': 'Región de Arica y Parinacota',
                'RegiÃ³n de TarapacÃ¡': 'Región de Tarapacá',
                'RegiÃ³n de Ã‘uble': 'Región de Ñuble',
                'RegiÃ³n de Los RÃ\xados': 'Región de Los Ríos',
                'RegiÃ³n de Magallanes y de la AntÃ¡rtica Chilena': 'Región de Magallanes y de la Antártica Chilena'
            }
            df['region'] = df['region'].replace(region_mapping)
        for col in df.select_dtypes(include=[np.number]).columns:
                df[col].fillna(df[col].mean(), inplace=True)

        for col in df.select_dtypes(include=[object]).columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
        print("Datos cargados correctamente.")
        #Columnas numericas
        numeric_cols = ['clientes_facturados', 'e1_kwh', 'e2_kwh', 'energia_kwh']
        for col in numeric_cols:
            # Usamos errors='coerce' para convertir cualquier valor no numérico a NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
        filas_iniciales = len(df)
        # Aplicar el filtro: conservar solo filas donde todas las columnas críticas son >= 0
        df_clean = df[(df['clientes_facturados'] >= 0) & 
                    (df['e1_kwh'] >= 0) & 
                    (df['e2_kwh'] >= 0) & 
                    (df['energia_kwh'] >= 0)].copy() # usamos .copy() para evitar warnings

        filas_borradas = filas_iniciales - len(df_clean)
        df = df_clean # Ahora df es el DataFrame limpio

        print(f"\Limpieza terminada , # de anomalias: {filas_borradas} filas eliminadas (valores < 0).")
        print(f"Total de filas limpias para modelado: {len(df)}")

        # Rellenar nulos en numéricas con la media
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col].fillna(df[col].mean(), inplace=True)

        # Rellenar nulos en categóricas con la moda
        for col in df.select_dtypes(include=[object]).columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{nombre_archivo}'.")
    except Exception as e:
        print(f"Ocurrió un error: {e}")

    return df