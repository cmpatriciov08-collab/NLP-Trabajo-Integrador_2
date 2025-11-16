# 🏗️ Arquitectura del Sistema RAG - Discursos de Javier Milei

## 🎯 Visión General del Proyecto

Este documento define la arquitectura completa del sistema RAG (Retrieval-Augmented Generation) para consultar y analizar los discursos públicos del presidente Javier Milei, cumpliendo con todos los requisitos del Trabajo Integrador 2.

## 📋 Requisitos del TP2 Cumplidos

### ✅ Requisitos Técnicos Obligatorios
1. **Sistema RAG funcional**: Pipeline completo (ingesta → embeddings → almacenamiento → recuperación → generación)
2. **Base de datos vectorial**: ChromaDB para almacenamiento y búsqueda semántica
3. **LangChain**: Orquestación del flujo RAG
4. **Modelo de lenguaje**: Gemini (via API)
5. **Interfaz Streamlit**: Aplicación conversacional funcional
6. **Corpus de documentos**: 10+ discursos de Javier Milei
7. **Deployment**: Hugging Face Spaces

### ✅ Requisitos de Documentación
8. **Repositorio GitHub**: Código fuente organizado
9. **README completo**: Documentación técnica completa
10. **Citación de fuentes**: Sistema de referencias implementado

## 🏛️ Arquitectura del Sistema

### Diagrama de Componentes

```
┌─────────────────────────────────────────────────────────────────┐
│                  RAG MILEI - SISTEMA COMPLETO                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   FRONTEND   │    │    BACKEND   │    │   STORAGE    │       │
│  │  Streamlit   │◄──►│  LangChain   │◄──►│   ChromaDB   │       │
│  │   Interface  │    │   + Gemini   │    │   Vector DB  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                    │                    │              │
│         ▼                    ▼                    ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   DOCUMENT   │    │   SECURITY   │    │    METRICS   │       │
│  │   PROCESSOR  │    │   MANAGER    │    │   ENGINE     │       │
│  │   (PDF/TXT)  │    │ (Rate Limit) │    │  (Analytics) │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 Pipeline RAG Completo

### Fase 1: Ingesta de Documentos (Offline)
```
Casa Rosada Web ──[Web Scraping]──► Discursos Raw ──[Limpieza]──► Texto Limpio
                                    │
                                    ▼
Text Splitter ──[Chunking]──► Fragmentos ──[Embeddings]──► Vectores
                                    │
                                    ▼
ChromaDB ──[Almacenamiento]──► Base Vectorial Persistente
```

### Fase 2: Consulta y Respuesta (Online)
```
Usuario Query ──[Streamlit]──► Embedding Query ──[Búsqueda]──► Top-K Chunks
                                        │
                                        ▼
Gemini LLM ──[Generación]──► Respuesta Contextual ──[Response]──► Usuario
                                        │
                                        ▼
Source Attribution ──[Citations]──► Document References ──[Display]──► UI
```

## 🛠️ Stack Tecnológico

| Componente | Tecnología | Versión | Propósito |
|------------|------------|---------|-----------|
| **Frontend** | Streamlit | 1.28+ | Interfaz web responsiva |
| **LLM** | Google Gemini | 1.5 Flash | Generación de respuestas |
| **Vector DB** | ChromaDB | 0.4+ | Almacenamiento de embeddings |
| **Embeddings** | Sentence Transformers | 2.2+ | Modelos multilenguaje |
| **Orquestación** | LangChain | 0.1+ | Pipeline RAG completo |
| **Document Processing** | PyPDF2, python-docx | Latest | Extracción de texto |
| **Web Scraping** | BeautifulSoup4 | 4.12+ | Scraping Casa Rosada |
| **Deployment** | Hugging Face Spaces | Latest | Hosting cloud |
| **Containerización** | Docker | Latest | Ambiente reproducible |

## 📁 Estructura del Proyecto

```
tp3/
├── 📄 app.py                      # Aplicación Streamlit principal
├── 📄 requirements.txt            # Dependencias del proyecto
├── 📄 README.md                   # Documentación principal
├── 📄 Dockerfile                  # Configuración Docker
├── 📄 .streamlit/
│   └── 📄 config.toml            # Configuración Streamlit
├── 📄 .env.example               # Template variables entorno
├── 📄 .gitignore                 # Archivos ignorados
├── 📁 src/                       # Código fuente principal
│   ├── 📄 __init__.py
│   ├── 📄 rag_system.py          # Sistema RAG core
│   ├── 📄 document_processor.py  # Procesador documentos
│   ├── 📄 web_scraper.py         # Scraper Casa Rosada
│   ├── 📄 embeddings_handler.py  # Manejador embeddings
│   ├── 📄 vector_store.py        # Interface ChromaDB
│   ├── 📄 llm_handler.py         # Handler Gemini
│   └── 📄 utils.py               # Utilidades
├── 📁 data/                      # Datos del proyecto
│   ├── 📁 corpus/                # Documentos fuente
│   ├── 📁 processed/             # Documentos procesados
│   └── 📁 vector_db/             # Base de datos vectorial
├── 📁 tests/                     # Tests automatizados
│   ├── 📄 test_rag_system.py
│   ├── 📄 test_document_processor.py
│   └── 📄 test_integration.py
├── 📁 docs/                      # Documentación técnica
│   ├── 📄 API_REFERENCE.md
│   ├── 📄 DEPLOYMENT_GUIDE.md
│   └── 📄 TROUBLESHOOTING.md
├── 📁 scripts/                   # Scripts utilitarios
│   ├── 📄 setup.sh              # Setup automático
│   ├── 📄 ingest_documents.py   # Ingesta masiva
│   └── 📄 generate_corpus.py    # Generación corpus
└── 📁 demos/                     # Demos y ejemplos
    ├── 📄 demo_queries.md       # Consultas de ejemplo
    └── 📄 video_demo.md         # Guía demo en video
```

## 🔐 Configuración de Seguridad

### Variables de Entorno
```bash
# API Configuration
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
GEMINI_TEMPERATURE=0.1

# RAG Configuration
RAG_TOP_K=4
SIMILARITY_THRESHOLD=0.7
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Security
RATE_LIMIT_REQUESTS=10
RATE_LIMIT_WINDOW=60
MAX_FILE_SIZE=10485760  # 10MB

# Deployment
DEBUG=False
ENVIRONMENT=production
HF_SPACES=True
```

### Rate Limiting
- **Límite**: 10 consultas por minuto por usuario
- **Ventana**: 60 segundos
- **Respuesta**: 429 Too Many Requests con mensaje educativo

## 📊 Métricas y Analytics

### Métricas Principales
- **Total Queries**: Contador de consultas procesadas
- **Response Time**: Tiempo promedio de respuesta
- **Success Rate**: Porcentaje de respuestas exitosas
- **Cache Hit Rate**: Efectividad del sistema de caché
- **Document Coverage**: Documentos más consultados

### Dashboard de Analytics
```python
# Métricas en tiempo real
- Consultas por hora/día
- Tiempo de respuesta promedio
- Tasa de éxito de consultas
- Documentos más populares
- Distribución de tipos de consulta
- Errores más comunes
```

## 🚀 Deployment Strategy

### Hugging Face Spaces (Recomendado)
1. **Preparación**: Código listo para production
2. **Configuración**: Secrets y variables de entorno
3. **Deployment**: Build automático desde GitHub
4. **Testing**: Validación end-to-end en producción
5. **Monitoring**: Logs y métricas de uso

### Alternativas de Deployment
- **Streamlit Cloud**: Hosting directo desde GitHub
- **Heroku**: Deployment con Docker
- **VPS/Dedicado**: Control total del entorno

## 🧪 Estrategia de Testing

### Tests Unitarios
- Procesamiento de documentos
- Generación de embeddings
- Búsqueda vectorial
- Integración LLM

### Tests de Integración
- Pipeline completo RAG
- Interfaz Streamlit
- Deployment workflow
- Performance bajo carga

### Tests End-to-End
- Consulta completa usuario
- Respuesta con fuentes
- Persistencia de datos
- Manejo de errores

## 📈 Optimizaciones de Performance

### Cache Strategy
- **Response Cache**: TTL de 1 hora para respuestas
- **Embedding Cache**: Cache persistente de embeddings
- **Document Cache**: Cache de documentos procesados

### Chunking Strategy
- **Tamaño óptimo**: 1000 caracteres por chunk
- **Overlap**: 200 caracteres para contexto
- **Separadores inteligentes**: Párrafos, oraciones, frases

### Vector Search Optimization
- **Índice optimizado**: Configuración ChromaDB para performance
- **Búsqueda híbrida**: Combinación similarity + MMR
- **Filtros metadata**: Búsqueda por fecha, tipo, fuente

## 🔍 Características Especiales

### Sistema de Citación
- **Referencias automáticas**: Cada respuesta incluye fuentes
- **Metadata rica**: Título, fecha, página, tipo de documento
- **Traceability**: Seguimiento completo del origen de información

### Interfaz Conversacional
- **Historial persistente**: Mantiene contexto de conversación
- **Follow-up queries**: Respuestas contextuales
- **Multi-turn conversations**: Conversaciones complejas

### Gestión de Corpus
- **Actualización automática**: Scraping periódico de nuevos discursos
- **Metadata automática**: Extracción de fechas, títulos, fuentes
- **Validación de calidad**: Verificación de contenido válido

## 📚 Documentación Técnica

### Documentos Requeridos
1. **README.md**: Guía completa de uso e instalación
2. **API_REFERENCE.md**: Documentación técnica de componentes
3. **DEPLOYMENT_GUIDE.md**: Guía paso a paso para deployment
4. **TROUBLESHOOTING.md**: Solución de problemas comunes
5. **ARCHITECTURE.md**: Documentación de arquitectura (este documento)

### Cumplimiento TP2
- ✅ Documentación completa y profesional
- ✅ Instrucciones de instalación reproducibles
- ✅ Ejemplos de uso con consultas reales
- ✅ Decisiones técnicas justificadas
- ✅ Limitaciones y mejoras futuras documentadas

## 🎯 Conclusión

Esta arquitectura cumple todos los requisitos del TP2 mientras proporciona un sistema RAG robusto, escalable y listo para producción. El diseño modular permite fácil mantenimiento y extensión futura, mientras que la documentación completa asegura reproducibilidad y comprensión del sistema.

La implementación final en `tp3/` será una versión optimizada y production-ready del sistema RAG para análisis de discursos de Javier Milei.