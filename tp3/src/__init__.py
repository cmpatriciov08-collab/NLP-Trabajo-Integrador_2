"""
Sistema RAG para Discursos de Javier Milei
==========================================

Sistema completo de Recuperación y Generación Aumentada (RAG) que permite consultar
y analizar los discursos públicos del presidente Javier Milei usando técnicas
avanzadas de IA y procesamiento de lenguaje natural.

Funcionalidades principales:
- Web scraping automatizado de discursos de la Casa Rosada
- Procesamiento inteligente de documentos con chunking optimizado
- Base de datos vectorial con ChromaDB para búsqueda semántica
- Sistema RAG completo con LangChain y Google Gemini
- Interfaz conversacional con Streamlit
- Citación automática de fuentes
- Analytics y métricas de performance
- Deployment listo para Hugging Face Spaces

Autor: MVP Sistema RAG Milei
Fecha: 2025-11-16
Versión: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Sistema RAG Milei"
__email__ = "mvp@rag-milei.com"

# Importaciones principales
from .rag_system import RAGSystem
from .document_processor import DocumentProcessor
from .web_scraper import WebScraper
from .embeddings_handler import EmbeddingsHandler
from .vector_store import VectorStore
from .llm_handler import LLMHandler
from .utils import Utils

__all__ = [
    "RAGSystem",
    "DocumentProcessor", 
    "WebScraper",
    "EmbeddingsHandler",
    "VectorStore",
    "LLMHandler",
    "Utils"
]