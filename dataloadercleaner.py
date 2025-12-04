import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def clean_data():
    """
    Limpia datos de facturación eléctrica para Deep Learning.
    Returns: (df_clean, encoders_dict, normalization_stats)
    """
    
    try:
        scriptDir = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(scriptDir, "DatosFacturas.csv")
        
        # Cargar datos
        df = pd.read_csv(file, sep=';', encoding='latin-1')
        df.columns = df.columns.str.replace('ï»¿', '').str.strip()
        
        # Corregir encoding de regiones
        region_mapping = {
            'RegiÃÂ³n del Libertador Gral. Bernardo Oâ€™Higgins': "Región de O'Higgins",
            'RegiÃÂ³n del BiobÃ\xado': 'Región del Biobío',
            'RegiÃÂ³n del Maule': 'Región del Maule',
            'RegiÃÂ³n de ValparaÃ\xadso': 'Región de Valparaíso',
            'RegiÃÂ³n Metropolitana de Santiago': 'Región Metropolitana',
            'RegiÃÂ³n de La AraucanÃ\xada': 'Región de La Araucanía',
            'RegiÃÂ³n de Los Lagos': 'Región de Los Lagos',
            'RegiÃÂ³n de Atacama': 'Región de Atacama',
            'RegiÃÂ³n AisÃ©n del Gral.Carlos IbÃ¡Ã±ez del Campo': 'Región de Aysén',
            'RegiÃÂ³n de Coquimbo': 'Región de Coquimbo',
            'RegiÃÂ³n de Antofagasta': 'Región de Antofagasta',
            'RegiÃÂ³n de Arica y Parinacota': 'Región de Arica y Parinacota',
            'RegiÃÂ³n de TarapacÃ¡': 'Región de Tarapacá',
            'RegiÃÂ³n de Ã‘uble': 'Región de Ñuble',
            'RegiÃÂ³n de Los RÃ\xados': 'Región de Los Ríos',
            'RegiÃÂ³n de Magallanes y de la AntÃ¡rtica Chilena': 'Región de Magallanes'
        }
        df['region'] = df['region'].replace(region_mapping)
        df['comuna'] = df['comuna'].astype(str).str.strip()
        
        # Convertir tipos
        numeric_cols = ['clientes_facturados', 'e1_kwh', 'e2_kwh', 'energia_kwh']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['anio'] = pd.to_numeric(df['anio'], errors='coerce').astype(int)
        df['mes'] = pd.to_numeric(df['mes'], errors='coerce').astype(int)
        
        # Rellenar nulos ANTES de crear features
        for col in numeric_cols + ['anio', 'mes']:
            df[col] = df[col].fillna(df[col].median())
        
        for col in df.select_dtypes(include=[object]).columns:
            if len(df[col].mode()) > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        # Eliminar valores negativos y duplicados
        df = df[
            (df['clientes_facturados'] >= 0) & 
            (df['e1_kwh'] >= 0) & 
            (df['e2_kwh'] >= 0) & 
            (df['energia_kwh'] >= 0)
        ].drop_duplicates().copy()
        
        # Crear features
        df['consumo_promedio_cliente'] = df['energia_kwh'] / (df['clientes_facturados'] + 1)
        df['es_residencial'] = (df['tipo_clientes'] == 'Residencial').astype(int)
        df['es_verano'] = df['mes'].isin([12, 1, 2, 3]).astype(int)
        df['proporcion_e1'] = df['e1_kwh'] / (df['energia_kwh'] + 1)
        df['trimestre'] = ((df['mes'] - 1) // 3 + 1).astype(int)
        
        try:
            df['fecha'] = pd.to_datetime(
                df[['anio', 'mes']].rename(columns={'anio': 'year', 'mes': 'month'}).assign(day=1)
            )
        except:
            df['fecha'] = pd.NaT
        
        # Codificar categóricas
        encoders = {}
        for col in ['region', 'comuna', 'tipo_clientes', 'tarifa']:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        
        # Normalizar
        normalization_stats = {}
        numeric_features = ['energia_kwh', 'clientes_facturados', 
                            'consumo_promedio_cliente', 'e1_kwh', 'e2_kwh']
        
        for col in numeric_features:
            mean_val = df[col].mean()
            std_val = df[col].std()
            normalization_stats[col] = {'mean': mean_val, 'std': std_val}
            df[f'{col}_norm'] = (df[col] - mean_val) / (std_val + 1e-8)
        
        # Resumen mínimo
        print(f"✅ Limpieza completada: {len(df):,} registros ({df['anio'].min()}-{df['anio'].max()})")
        
        return df, encoders, normalization_stats
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None, None


def prepare_for_tensorflow(df):
    """Prepara X, y para TensorFlow"""
    feature_cols = [
        'anio', 'mes', 'trimestre', 'es_verano',
        'region_encoded', 'comuna_encoded', 'tarifa_encoded',
        'clientes_facturados', 'e1_kwh', 'e2_kwh',
        'consumo_promedio_cliente', 'proporcion_e1', 'es_residencial'
    ]
    
    X = df[feature_cols].values
    y = df['energia_kwh'].values  # ✅ CORRECCIÓN: Asegurar que sea numpy array
    
    return X, y, feature_cols  # ✅ CORRECCIÓN: Retornar los 3 valores


if __name__ == "__main__":
    df_clean, encoders, norm_stats = clean_data()
    
    if df_clean is not None:
        output_file = os.path.join(os.path.dirname(__file__), "datos_limpios.csv")
        df_clean.to_csv(output_file, index=False, encoding='utf-8')
        
        X, y, features = prepare_for_tensorflow(df_clean)
        print(f"📊 X: {X.shape} | y: {y.shape} | Features: {len(features)}")