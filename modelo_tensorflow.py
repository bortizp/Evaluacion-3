import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from dataloadercleaner import clean_data, prepare_for_tensorflow

# Cargar y preparar datos
print("🔄 Cargando datos...")
df_clean, encoders, norm_stats = clean_data()
X, y, features = prepare_for_tensorflow(df_clean)

# División train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalizar y (target) para mejor convergencia
y_mean = y_train.mean()
y_std = y_train.std()
y_train_norm = (y_train - y_mean) / y_std
y_test_norm = (y_test - y_mean) / y_std

print(f"📊 Train: {X_train.shape} | Test: {X_test.shape}")

# Arquitectura del modelo mejorada
model = keras.Sequential([
    keras.layers.Input(shape=[X_train.shape[1]]),
    keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)  # Regresión: predecir consumo energético
])

# Compilar con learning rate adaptativo
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Callbacks para mejor entrenamiento
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001
)

# Entrenar
print("\n🚀 Entrenando modelo...")
history = model.fit(
    X_train, y_train_norm,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluar
print("\n📈 Evaluando modelo...")
y_pred_norm = model.predict(X_test, verbose=0)
y_pred = y_pred_norm * y_std + y_mean  # Desnormalizar predicciones

# Métricas
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = np.mean((y_test - y_pred.flatten())**2)

print(f"\n✅ Resultados:")
print(f"   - MAE: {mae:,.0f} kWh")
print(f"   - R²: {r2:.4f}")
print(f"   - MSE: {mse:,.0f}")

# Visualizar curvas de entrenamiento
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Evolución de la Pérdida')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Evolución del Error Absoluto Medio')
plt.xlabel('Epoch')
plt.ylabel('MAE (kWh)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('entrenamiento_modelo.png', dpi=300)
plt.show()

# Predicción vs Real
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, s=10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Consumo Real (kWh)')
plt.ylabel('Consumo Predicho (kWh)')
plt.title(f'Predicción vs Realidad (R² = {r2:.4f})')
plt.grid(True)
plt.tight_layout()
plt.savefig('prediccion_vs_real.png', dpi=300)
plt.show()

# Guardar modelo
model.save('modelo_consumo_energetico.keras')
print("\n💾 Modelo guardado como 'modelo_consumo_energetico.keras'")