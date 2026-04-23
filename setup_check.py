"""
setup_check.py — Verifica que todo está listo para ejecutar el chatbot
"""
import sys, importlib

REQUIRED = [
    ("dash", "dash"),
    ("dash_bootstrap_components", "dash-bootstrap-components"),
    ("plotly", "plotly"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("sentence_transformers", "sentence-transformers"),
    ("faiss", "faiss-cpu"),
    ("transformers", "transformers"),
    ("datasets", "datasets"),
    ("sklearn", "scikit-learn"),
    ("evaluate", "evaluate"),
    ("accelerate", "accelerate"),
]

OPTIONAL = [
    ("anthropic", "anthropic"),
    ("openai", "openai"),
]

print("\n" + "="*55)
print("  MúsicBot — Verificación de dependencias")
print("="*55)

all_ok = True
for mod, pkg in REQUIRED:
    try:
        importlib.import_module(mod)
        print(f" {pkg}")
    except ImportError:
        print(f" {pkg}  →  pip install {pkg}")
        all_ok = False

print("\n  Opcionales (APIs externas):")
for mod, pkg in OPTIONAL:
    try:
        importlib.import_module(mod)
        print(f"  {pkg}")
    except ImportError:
        print(f" {pkg}  (no instalado — modo local OK)")

print("\n" + "="*55)
if all_ok:
    print("  Todo listo. Ejecuta:")
    print("     python app/chatbot_app.py")
else:
    print("     Instala las dependencias faltantes:")
    print("     pip install -r requirements.txt")
    print("     pip install torch")
print("="*55 + "\n")
