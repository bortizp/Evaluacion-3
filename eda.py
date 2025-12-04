import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos limpios
df = pd.read_csv('datos_limpios.csv')

# Configuración visual
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# 1. Evolución temporal del consumo
plt.figure()
consumo_anual = df.groupby('anio')['energia_kwh'].sum() / 1e6
consumo_anual.plot(kind='bar', color='steelblue')
plt.title('Consumo Energético Total por Año (GWh)')
plt.ylabel('Energía (GWh)')
plt.xlabel('Año')
plt.tight_layout()
plt.savefig('consumo_anual.png', dpi=300)
plt.show()

# 2. Top 5 regiones con mayor consumo
plt.figure()
top_regiones = df.groupby('region')['energia_kwh'].sum().nlargest(5) / 1e6
top_regiones.plot(kind='barh', color='coral')
plt.title('Top 5 Regiones con Mayor Consumo')
plt.xlabel('Energía (GWh)')
plt.tight_layout()
plt.savefig('top_regiones.png', dpi=300)
plt.show()

# 3. Residencial vs No Residencial
plt.figure()
consumo_tipo = df.groupby('tipo_clientes')['energia_kwh'].sum() / 1e6
consumo_tipo.plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Distribución Residencial vs No Residencial')
plt.ylabel('')
plt.tight_layout()
plt.savefig('residencial_vs_noResidencial.png', dpi=300)
plt.show()

# 4. Heatmap de correlación
plt.figure(figsize=(10, 8))
numeric_cols = ['anio', 'mes', 'clientes_facturados', 'e1_kwh', 'e2_kwh', 
                'energia_kwh', 'consumo_promedio_cliente', 'es_residencial']
# Filtrar solo las columnas que existen
numeric_cols = [col for col in numeric_cols if col in df.columns]
correlation = df[numeric_cols].corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Matriz de Correlación')
plt.tight_layout()
plt.savefig('correlacion.png', dpi=300)
plt.show()

print("✅ Gráficos guardados exitosamente")