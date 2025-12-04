import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import sqlite3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Suponemos que load_data_from_sql() ahora carga TODAS las columnas necesarias
def load_data_for_model(db_name: str = 'proyecto_nn.db', table_name: str = 'datos_limpios'):
    """
    Carga todas las columnas necesarias, incluyendo las categóricas.
    """
    conn = sqlite3.connect(db_name)
    sql_query = f"""
    SELECT 
        clientes_facturados, 
        region, 
        tipo_clientes, 
        tarifa,
        energia_kwh 
    FROM {table_name}
    LIMIT 500000;
    """
    df = pd.read_sql_query(sql_query, conn)
    conn.close()
    return df

# -------------------------------------------------------------------------
# AJUSTE DEL PREPROCESAMIENTO Y MODELADO
# -------------------------------------------------------------------------

df_modelado = load_data_for_model()

# 1. One-Hot Encoding para variables categóricas
# Esto convierte columnas categóricas (como 'region') en columnas binarias (ej: region_A, region_B)
df_modelado_encoded = pd.get_dummies(df_modelado, columns=['region', 'tipo_clientes', 'tarifa'], drop_first=True)

# 2. Separar X (Features) y y (Target)
y = df_modelado_encoded['energia_kwh'].values
# Excluimos la columna objetivo del conjunto X
X = df_modelado_encoded.drop(columns=['energia_kwh']).values
# Guardar las columnas de FEATURES para usar en la predicción en vivo (¡NUEVO!)
columnas_de_features = df_modelado_encoded.drop(columns=['energia_kwh']).columns.tolist() 
# 3. Estandarización de las características
# (Solo es necesario escalar las columnas numéricas si no usamos get_dummies, 
# pero escalar todo el conjunto de X es una práctica segura en el MLP)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. División de conjuntos
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 5. Definición y Entrenamiento del Modelo MLP (Red Neuronal)

# El input_shape ahora es el número de todas las variables (numéricas + categóricas codificadas)
input_dim = X_train.shape[1] 

model = Sequential([
    # Input Layer: Ajustamos el 'input_shape' al nuevo número de features
    Dense(128, activation='relu', input_shape=(input_dim,), name='Capa_Oculta_1'), 
    Dropout(0.2), 
    Dense(64, activation='relu', name='Capa_Oculta_2'),
    Dense(32, activation='relu', name='Capa_Oculta_3'),
    # Output Layer for Regression: 1 neurona lineal
    Dense(1, name='Capa_Salida_Prediccion') 
])

# Compilación y Entrenamiento
model.compile(loss='mse', optimizer='adam', metrics=['mae']) 

print("\n Iniciando Entrenamiento con Variables de Contexto...")
history = model.fit(
    X_train, y_train,
    epochs=10, # Aumentamos las epochs para que aprenda las relaciones más complejas
    batch_size=64,
    validation_split=0.1, 
    verbose=1
)

# 6. Evaluación (Para el Punto 3 de tu proyecto)
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nEvaluación Final - Pérdida (MSE): {loss:.2f}, MAE: {mae:.2f}")

# Asumimos que 'history', 'model', 'X_test', 'y_test', 'scaler', 
# y 'df_modelado_encoded' de la Sección 2 están disponibles.

# -------------------------------------------------------------------------
# CÓDIGO PARA VISUALIZAR LAS CURVAS DE APRENDIZAJE
# -------------------------------------------------------------------------

# 1. Gráfico de la Pérdida (Loss: MSE)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Pérdida (Entrenamiento)')
plt.plot(history.history['val_loss'], label='Pérdida (Validación)')
plt.title('Curva de Pérdida (Mean Squared Error)')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)

# 2. Gráfico de la Métrica (MAE)
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='MAE (Entrenamiento)')
plt.plot(history.history['val_mae'], label='MAE (Validación)')
plt.title('Curva de Error Absoluto (MAE)')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# FUNCIÓN DE PREDICCIÓN EN VIVO CON VARIABLES CATEGÓRICAS

def predecir_consumo_en_vivo(
    clientes: int, 
    region: str, 
    tipo_cliente: str, 
    tarifa: str,
    scaler, # El objeto StandardScaler entrenado
    model, # El modelo Keras entrenado
    df_train_columns: list # La lista de columnas guardada arriba
):
    """
    Realiza una predicción en vivo para un solo punto de datos, asegurando 
    que el preprocesamiento de One-Hot Encoding coincida con el entrenamiento.
    """
    
    # 1. Crear el DataFrame del nuevo dato
    nuevo_dato = pd.DataFrame([{
        'clientes_facturados': clientes, 
        'region': region, 
        'tipo_clientes': tipo_cliente, 
        'tarifa': tarifa
    }])
    
    # 2. Aplicar One-Hot Encoding
    dato_encoded = pd.get_dummies(nuevo_dato)
    
    # 3. Re-indexar para asegurar que tiene TODAS las columnas del entrenamiento
    # Las columnas nuevas que no existen en el nuevo_dato se rellenan con 0, 
    # y las columnas que sí existen mantienen su valor (1)
    dato_final = dato_encoded.reindex(columns=df_train_columns, fill_value=0)
    
    # 4. Escalar los datos (usando el scaler entrenado)
    dato_escalado = scaler.transform(dato_final.values)
    
    # 5. Predecir
    prediccion_array = model.predict(dato_escalado)
    prediccion_kwh = prediccion_array[0][0]
    
    return prediccion_kwh

    # -------------------------------------------------------------------------
# CÓDIGO DE LA DEMOSTRACIÓN (Sección 3)
# -------------------------------------------------------------------------

# DATOS DE PRUEBA EN VIVO:
# Elegir un escenario para predecir
clientes_facturados_test = 150 
region_test = 'Región del Biobío' 
tipo_cliente_test = 'Res' # Residencial
tarifa_test = 'T1'

# Ejecutar la predicción
prediccion = predecir_consumo_en_vivo(
    clientes_facturados_test, 
    region_test, 
    tipo_cliente_test, 
    tarifa_test,
    scaler, # De la sección 2
    model,  # De la sección 2
    columnas_de_features # La lista de columnas que guardaste
)

print("\n---------------------------------------------------------")
print("  DEMOSTRACIÓN DE PREDICCIÓN EN VIVO (SIN e1_kwh / e2_kwh)")
print("---------------------------------------------------------")
print(f"**Entradas de Contexto:**")
print(f" - Clientes Facturados: {clientes_facturados_test}")
print(f" - Región: {region_test}")
print(f" - Tipo de Cliente: {tipo_cliente_test}")
print(f" - Tarifa: {tarifa_test}")
print("---------------------------------------------------------")
print(f"**Predicción de Energía (kWh):** {prediccion:,.2f} kWh")
print("---------------------------------------------------------")