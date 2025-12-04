import pandas as pd
from dataLoaderCleaner import clean_data

def main():
    # Llamar a la función de limpieza de datos
    df_clean = clean_data()
    
    # Mostrar las primeras filas del DataFrame limpio
    print(df_clean.head())



main()