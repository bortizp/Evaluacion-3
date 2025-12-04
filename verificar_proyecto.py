"""
Script de Verificación del Proyecto
Valida que todos los componentes estén funcionando correctamente
"""

import os
import sys

print("="*60)
print("🔍 VERIFICACIÓN DEL PROYECTO - EVALUACIÓN 3")
print("="*60)

# Lista de archivos requeridos
archivos_requeridos = {
    "Scripts Python": [
        "dataloadercleaner.py",
        "eda.py",
        "modelo_tensorflow.py",
        "sql_integration.py"
    ],
    "Datos": [
        "DatosFacturas.csv",
        "datos_limpios.csv",
        "facturacion_electrica.db"
    ],
    "Modelo": [
        "modelo_consumo_energetico.keras"
    ],
    "Visualizaciones": [
        "consumo_anual.png",
        "top_regiones.png",
        "residencial_vs_noResidencial.png",
        "correlacion.png",
        "entrenamiento_modelo.png",
        "prediccion_vs_real.png"
    ],
    "Documentación": [
        "README.md",
        "RESUMEN_PROYECTO.md"
    ]
}

# Verificar existencia de archivos
print("\n📂 Verificando archivos...")
total_archivos = 0
archivos_encontrados = 0

for categoria, archivos in archivos_requeridos.items():
    print(f"\n{categoria}:")
    for archivo in archivos:
        total_archivos += 1
        existe = os.path.exists(archivo)
        archivos_encontrados += existe
        
        icono = "✅" if existe else "❌"
        tam = f"{os.path.getsize(archivo)/1024:.1f} KB" if existe else "N/A"
        print(f"  {icono} {archivo:<40} {tam:>15}")

# Resumen
print("\n" + "="*60)
print(f"📊 RESUMEN: {archivos_encontrados}/{total_archivos} archivos encontrados")

if archivos_encontrados == total_archivos:
    print("✅ PROYECTO COMPLETO - Listo para entregar!")
else:
    print(f"⚠️  FALTAN {total_archivos - archivos_encontrados} archivos")

# Verificar librerías
print("\n" + "="*60)
print("📦 Verificando librerías instaladas...")

librerias = [
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "sklearn",
    "tensorflow"
]

for lib in librerias:
    try:
        __import__(lib)
        print(f"  ✅ {lib}")
    except ImportError:
        print(f"  ❌ {lib} (ejecuta: pip install {lib})")

print("\n" + "="*60)
print("🎯 CHECKLIST FINAL")
print("="*60)

checklist = [
    ("Datos limpios generados", os.path.exists("datos_limpios.csv")),
    ("Modelo entrenado", os.path.exists("modelo_consumo_energetico.keras")),
    ("Base de datos SQL", os.path.exists("facturacion_electrica.db")),
    ("Gráficos de EDA (4)", sum([os.path.exists(f) for f in ["consumo_anual.png", "top_regiones.png", "residencial_vs_noResidencial.png", "correlacion.png"]]) == 4),
    ("Gráficos del modelo (2)", sum([os.path.exists(f) for f in ["entrenamiento_modelo.png", "prediccion_vs_real.png"]]) == 2),
    ("Documentación completa", os.path.exists("README.md") and os.path.exists("RESUMEN_PROYECTO.md"))
]

for tarea, completada in checklist:
    icono = "✅" if completada else "❌"
    print(f"  {icono} {tarea}")

# Resultado final
print("\n" + "="*60)
if all([c[1] for c in checklist]):
    print("🎉 ¡FELICIDADES! EL PROYECTO ESTÁ COMPLETO")
    print("\n📝 Próximos pasos:")
    print("  1. Revisar README.md y RESUMEN_PROYECTO.md")
    print("  2. Grabar video de 10 minutos")
    print("  3. Subir todo a GitHub/Drive")
    print("  4. Entregar enlace al profesor")
else:
    print("⚠️  Completa los elementos faltantes antes de entregar")

print("="*60)
