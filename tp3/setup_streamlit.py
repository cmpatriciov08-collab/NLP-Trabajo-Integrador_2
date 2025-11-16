"""
Setup Script para Sistema RAG - Discursos de Javier Milei
========================================================

Script de inicialización y configuración del sistema RAG.
Configura el entorno, instala dependencias y inicializa el sistema.

Uso:
    python setup_streamlit.py
    python setup_streamlit.py --init-only    # Solo inicializar sistema
    python setup_streamlit.py --check        # Verificar configuración
    python setup_streamlit.py --demo         # Ejecutar demo

Autor: Sistema RAG Milei
Fecha: 2025-11-16
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_python_version():
    """Verificar versión de Python."""
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8+ requerido. Versión actual: %s", sys.version)
        return False
    logger.info("✅ Python %s", sys.version.split()[0])
    return True


def create_directories():
    """Crear estructura de directorios necesaria."""
    directories = [
        "data",
        "data/cache",
        "data/cache/embeddings", 
        "data/cache/responses",
        "data/vector_db",
        "data/temp",
        "data/corpus",
        "data/backups",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info("📁 Directorio creado: %s", directory)
    
    return True


def install_dependencies():
    """Instalar dependencias del sistema."""
    logger.info("📦 Instalando dependencias...")
    
    # Actualizar pip
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        logger.info("✅ pip actualizado")
    except subprocess.CalledProcessError as e:
        logger.warning("⚠️ Error actualizando pip: %s", e)
    
    # Instalar dependencias principales
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            logger.info("✅ Dependencias instaladas desde requirements.txt")
        except subprocess.CalledProcessError as e:
            logger.error("❌ Error instalando dependencias: %s", e)
            return False
    else:
        # Instalar dependencias manualmente
        dependencies = [
            "streamlit>=1.28.0",
            "langchain>=0.1.0",
            "langchain-google-genai>=2.0.0", 
            "langchain-chroma>=0.1.0",
            "chromadb>=0.4.0",
            "sentence-transformers>=2.2.0",
            "sentencepiece>=0.1.99",
            "transformers>=4.35.0",
            "torch>=2.0.0",
            "beautifulsoup4>=4.12.0",
            "requests>=2.31.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "plotly>=5.15.0",
            "python-docx>=1.0.0",
            "pypdf>=3.0.0",
            "streamlit-option-menu>=0.3.0",
            "python-dotenv>=1.0.0",
            "psutil>=5.9.0"
        ]
        
        for dep in dependencies:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                logger.info("✅ Instalado: %s", dep)
            except subprocess.CalledProcessError as e:
                logger.warning("⚠️ Error instalando %s: %s", dep, e)
    
    return True


def create_env_file():
    """Crear archivo .env de ejemplo."""
    env_content = """# Configuración del Sistema RAG - Discursos de Javier Milei
# Copiar este archivo como .env y completar con los valores reales

# =============================================================================
# CONFIGURACIÓN DE GOOGLE GEMINI
# =============================================================================
# Obtener API key en: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY=tu_api_key_de_google_aqui

# =============================================================================
# CONFIGURACIÓN DE EMBEDDINGS
# =============================================================================
# Modelo de embeddings (recomendado para español)
EMBEDDINGS_MODEL=intfloat/multilingual-e5-large

# =============================================================================
# CONFIGURACIÓN DE LLM
# =============================================================================
# Modelo de Gemini (gemini-1.5-flash es más rápido, gemini-1.5-pro más potente)
LLM_MODEL=gemini-1.5-flash
GEMINI_TEMPERATURE=0.1
GEMINI_MAX_TOKENS=2000
GEMINI_TOP_P=0.8
GEMINI_TOP_K=40

# =============================================================================
# CONFIGURACIÓN DE CHROMADB
# =============================================================================
# Directorio para persistencia
VECTOR_DB_PATH=data/vector_db
CHROMA_COLLECTION_NAME=milei_discursos

# =============================================================================
# CONFIGURACIÓN DE PERFORMANCE
# =============================================================================
# Configuración de embeddings
EMBEDDINGS_BATCH_SIZE=32
EMBEDDINGS_CACHE_ENABLED=true

# Configuración de LLM
LLM_RATE_LIMIT_CALLS_PER_MINUTE=10
LLM_CACHE_ENABLED=true

# =============================================================================
# CONFIGURACIÓN DE CHUNKING
# =============================================================================
# Parámetros para división de documentos
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MIN_CHUNK_SIZE=100
MAX_CHUNK_SIZE=2000

# =============================================================================
# CONFIGURACIÓN DE SEARCH
# =============================================================================
# Parámetros de búsqueda
RAG_TOP_K=4
SIMILARITY_THRESHOLD=0.7
SEARCH_TYPE=similarity

# =============================================================================
# CONFIGURACIÓN DE STREAMLIT
# =============================================================================
# Puerto para la aplicación
STREAMLIT_PORT=8501
STREAMLIT_HOST=0.0.0.0

# Configuración de logging
LOG_LEVEL=INFO
LOG_FILE=logs/rag_system.log

# =============================================================================
# CONFIGURACIÓN DE DEBUG
# =============================================================================
# Habilitar modo debug
DEBUG=false
VERBOSE=false
"""
    
    env_file = Path(".env.example")
    with open(env_file, 'w', encoding='utf-8') as f:
        f.write(env_content)
    
    logger.info("📄 Archivo .env.example creado: %s", env_file)
    return True


def create_streamlit_secrets():
    """Crear archivo de secrets para Streamlit Cloud."""
    secrets_content = """# Streamlit Secrets Configuration
# Para usar en Hugging Face Spaces o Streamlit Cloud

[GOOGLE_API_KEY]
value = "tu_api_key_aqui"

[EMBEDDINGS_CONFIG]
model_name = "intfloat/multilingual-e5-large"
batch_size = 32

[LLM_CONFIG] 
model_name = "gemini-1.5-flash"
temperature = 0.1
max_tokens = 2000
top_p = 0.8
top_k = 40

[RAG_CONFIG]
top_k = 4
similarity_threshold = 0.7
search_type = "similarity"

[CHUNKING_CONFIG]
chunk_size = 1000
chunk_overlap = 200

[VECTOR_DB_CONFIG]
persist_directory = "data/vector_db"
collection_name = "milei_discursos"

[PERFORMANCE_CONFIG]
embeddings_cache_enabled = true
llm_cache_enabled = true
rate_limit_calls_per_minute = 10
"""
    
    secrets_file = Path(".streamlit/secrets.toml")
    secrets_file.parent.mkdir(exist_ok=True)
    
    with open(secrets_file, 'w', encoding='utf-8') as f:
        f.write(secrets_content)
    
    logger.info("🔑 Archivo .streamlit/secrets.toml creado: %s", secrets_file)
    return True


def initialize_rag_system():
    """Inicializar el sistema RAG."""
    logger.info("🚀 Inicializando sistema RAG...")
    
    try:
        # Importar y inicializar el sistema
        sys.path.append(str(Path(__file__).parent))
        from src.rag_system import RAGSystem
        
        # Crear instancia del sistema
        rag_system = RAGSystem()
        
        # Inicializar componentes
        init_result = rag_system.initialize()
        
        if init_result.get('success'):
            logger.info("✅ Sistema RAG inicializado correctamente")
            logger.info("📊 Documentos indexados: %s", init_result.get('documents_indexed', 0))
            logger.info("🧠 Modelo embeddings: %s", init_result.get('embeddings_model', 'N/A'))
            logger.info("🤖 Modelo LLM: %s", init_result.get('llm_model', 'N/A'))
            return True
        else:
            logger.error("❌ Error inicializando sistema RAG: %s", init_result.get('error'))
            return False
            
    except Exception as e:
        logger.error("❌ Error inesperado inicializando RAG: %s", e)
        return False


def run_demo():
    """Ejecutar demo del sistema."""
    logger.info("🎬 Iniciando demo del sistema RAG...")
    
    try:
        # Importar y probar componentes
        sys.path.append(str(Path(__file__).parent))
        from src.rag_system import RAGSystem
        from src.utils import Utils
        
        utils = Utils()
        
        # Crear sistema
        rag_system = RAGSystem()
        init_result = rag_system.initialize()
        
        if not init_result.get('success'):
            logger.error("❌ No se puede ejecutar demo sin inicializar el sistema")
            return False
        
        # Ejecutar consultas de prueba
        demo_queries = [
            "¿Cuáles son las principales políticas económicas de Milei?",
            "¿Qué dice sobre la inflación?",
            "¿Cuáles son sus propuestas sobre el déficit fiscal?"
        ]
        
        for i, query in enumerate(demo_queries, 1):
            logger.info("🔍 Demo %s: %s", i, query)
            
            try:
                result = rag_system.query(query, include_sources=False)
                
                if result.get('success'):
                    answer = result.get('answer', 'Sin respuesta')
                    print(f"🤖 Respuesta: {answer[:200]}...")
                    print(f"⏱️ Tiempo: {result.get('processing_time_seconds', 0):.2f}s")
                    print(f"🎯 Confianza: {result.get('confidence', 0):.1%}")
                    print("-" * 80)
                else:
                    logger.error("❌ Error en consulta: %s", result.get('error'))
                    
            except Exception as e:
                logger.error("❌ Error en demo query: %s", e)
        
        logger.info("✅ Demo completado")
        return True
        
    except Exception as e:
        logger.error("❌ Error ejecutando demo: %s", e)
        return False


def run_streamlit():
    """Ejecutar aplicación Streamlit."""
    logger.info("🚀 Iniciando aplicación Streamlit...")
    
    app_file = Path(__file__).parent / "app.py"
    
    if not app_file.exists():
        logger.error("❌ Archivo app.py no encontrado: %s", app_file)
        return False
    
    try:
        # Verificar si está configurada la API key
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key or api_key == 'tu_api_key_de_google_aqui':
            logger.warning("⚠️ GOOGLE_API_KEY no configurada. Configure el archivo .env")
        
        # Ejecutar Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", str(app_file),
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false"
        ]
        
        logger.info("🌐 Ejecutando: %s", " ".join(cmd))
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        logger.info("👋 Aplicación detenida por el usuario")
    except Exception as e:
        logger.error("❌ Error ejecutando Streamlit: %s", e)
        return False
    
    return True


def check_configuration():
    """Verificar configuración del sistema."""
    logger.info("🔍 Verificando configuración del sistema...")
    
    checks = {
        "Python": check_python_version(),
        "Directorios": create_directories(),
        "Archivo .env": os.path.exists(".env.example"),
        "App Streamlit": os.path.exists("app.py"),
        "Código fuente": all([
            os.path.exists("src/rag_system.py"),
            os.path.exists("src/document_processor.py"),
            os.path.exists("src/web_scraper.py"),
            os.path.exists("src/embeddings_handler.py"),
            os.path.exists("src/vector_store.py"),
            os.path.exists("src/llm_handler.py"),
            os.path.exists("src/utils.py")
        ])
    }
    
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        logger.info("%s %s", status, check)
    
    # Verificar variables de entorno
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key and api_key != 'tu_api_key_de_google_aqui':
        logger.info("✅ GOOGLE_API_KEY configurada")
    else:
        logger.warning("⚠️ GOOGLE_API_KEY no configurada (opcional para desarrollo)")
    
    all_passed = all(checks.values())
    if all_passed:
        logger.info("🎉 Sistema configurado correctamente")
    else:
        logger.error("❌ Configuración incompleta")
    
    return all_passed


def main():
    """Función principal del setup."""
    parser = argparse.ArgumentParser(
        description="Setup del Sistema RAG - Discursos de Javier Milei",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python setup_streamlit.py              # Setup completo
  python setup_streamlit.py --init-only  # Solo inicializar sistema
  python setup_streamlit.py --check      # Verificar configuración
  python setup_streamlit.py --demo       # Ejecutar demo
  python setup_streamlit.py --run        # Ejecutar Streamlit
        """
    )
    
    parser.add_argument("--init-only", action="store_true",
                       help="Solo inicializar sistema RAG")
    parser.add_argument("--check", action="store_true",
                       help="Verificar configuración")
    parser.add_argument("--demo", action="store_true",
                       help="Ejecutar demo del sistema")
    parser.add_argument("--run", action="store_true",
                       help="Ejecutar aplicación Streamlit")
    parser.add_argument("--force", action="store_true",
                       help="Forzar reinstalación de dependencias")
    
    args = parser.parse_args()
    
    # Banner
    logger.info("🇦🇷" + "="*60)
    logger.info("    SISTEMA RAG - DISCURSOS DE JAVIER MILEI")
    logger.info("    Setup e Inicialización del Sistema")
    logger.info("    IFTS 24 - Procesamiento del Habla e Introducción a LLMs")
    logger.info("="*61)
    
    success = True
    
    # Verificar Python
    if not check_python_version():
        return 1
    
    # Verificar configuración
    if args.check:
        success = check_configuration()
        return 0 if success else 1
    
    # Crear directorios y archivos de configuración
    if not args.init_only:
        create_directories()
        create_env_file()
        create_streamlit_secrets()
        
        if not args.demo and not args.run:
            install_dependencies()
    
    # Inicializar sistema RAG
    if args.init_only or args.demo:
        success = initialize_rag_system()
        if not success:
            logger.error("❌ No se pudo inicializar el sistema RAG")
            return 1
    
    # Ejecutar demo
    if args.demo:
        success = run_demo()
        return 0 if success else 1
    
    # Ejecutar Streamlit
    if args.run:
        success = run_streamlit()
        return 0 if success else 1
    
    # Setup completo
    if not args.init_only and not args.demo and not args.run:
        logger.info("✅ Setup completado exitosamente")
        logger.info("")
        logger.info("📋 PRÓXIMOS PASOS:")
        logger.info("1. Configurar API key en archivo .env")
        logger.info("2. Ejecutar: python setup_streamlit.py --init-only")
        logger.info("3. Ejecutar: python setup_streamlit.py --run")
        logger.info("")
        logger.info("🌐 La aplicación estará disponible en: http://localhost:8501")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())