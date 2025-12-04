# 🔌 Predicción de Consumo Energético en Chile

## 📌 Descripción del Proyecto

Análisis y predicción del consumo eléctrico en Chile (2015-2024) usando:

- **Python** para limpieza de datos
- **SQL** (SQLite) para consultas analíticas
- **TensorFlow/Keras** para Deep Learning

---

## 📂 Estructura del Proyecto

```
Evaluacion-3/
├── DatosFacturas.csv          # Datos originales (CNE)
├── datos_limpios.csv          # Datos procesados
├── dataloadercleaner.py       # Limpieza y preparación
├── eda.py                     # Análisis exploratorio
├── modelo_tensorflow.py       # Modelo de Deep Learning
├── sql_integration.py         # Integración con SQL
├── facturacion_electrica.db   # Base de datos SQLite
└── README.md                  # Este archivo
```

---

## 🚀 Instalación

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

---

## 📊 Ejecución

### 1. Limpiar datos

```bash
python dataloadercleaner.py
```

### 2. Análisis exploratorio

```bash
python eda.py
```

### 3. Entrenar modelo

```bash
python modelo_tensorflow.py
```

### 4. Consultas SQL

```bash
python sql_integration.py
```

---

## 🎯 Resultados Obtenidos

### Datos Procesados

- **Registros totales**: 486,610
- **Rango temporal**: 2015 - 2024
- **Regiones únicas**: 16
- **Comunas únicas**: 330
- **Energía facturada**: 262.3 TWh
- **Clientes totales**: 710 millones

### Rendimiento del Modelo

- **R² Score**: 0.9191 (91.91% de precisión) ✅
- **MAE**: 234,695 kWh
- **MSE**: 418,917,056,500
- **Epochs**: 12 (con early stopping)

### Top 5 Comunas con Mayor Consumo

1. Santiago - 12.3 TWh
2. Las Condes - 9.3 TWh
3. Maipú - 6.7 TWh
4. Providencia - 5.8 TWh
5. Antofagasta - 5.1 TWh

---

## 👥 Autores

- Bastian [Apellido]
- [Compañero 2]
- [Compañero 3]

---

## 📅 Fecha

Diciembre 2024
