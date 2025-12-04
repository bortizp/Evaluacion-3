import pandas as pd
import sqlite3

# Cargar datos limpios
df = pd.read_csv('datos_limpios.csv')

# Crear base de datos SQLite
conn = sqlite3.connect('facturacion_electrica.db')

# Insertar datos en tabla
df.to_sql('facturacion', conn, if_exists='replace', index=False)

print("✅ Base de datos SQLite creada: 'facturacion_electrica.db'")

# Ejemplo de consultas SQL
print("\n📊 Consultas SQL:")

# 1. Top 10 comunas con mayor consumo
query1 = """
SELECT comuna, SUM(energia_kwh) as consumo_total
FROM facturacion
GROUP BY comuna
ORDER BY consumo_total DESC
LIMIT 10
"""
resultado1 = pd.read_sql_query(query1, conn)
print("\n🏆 Top 10 Comunas con Mayor Consumo:")
print(resultado1)

# 2. Consumo promedio por región y año
query2 = """
SELECT region, anio, AVG(energia_kwh) as consumo_promedio
FROM facturacion
GROUP BY region, anio
ORDER BY region, anio
"""
resultado2 = pd.read_sql_query(query2, conn)
print("\n📍 Consumo Promedio por Región y Año:")
print(resultado2.head(20))

# 3. Clientes residenciales vs no residenciales
query3 = """
SELECT tipo_clientes, 
       COUNT(*) as registros,
       SUM(clientes_facturados) as total_clientes,
       SUM(energia_kwh) as energia_total
FROM facturacion
GROUP BY tipo_clientes
"""
resultado3 = pd.read_sql_query(query3, conn)
print("\n👥 Estadísticas por Tipo de Cliente:")
print(resultado3)

conn.close()
print("\n✅ Conexión cerrada")