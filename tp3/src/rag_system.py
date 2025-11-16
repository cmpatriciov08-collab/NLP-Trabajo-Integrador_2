"""
Sistema RAG Principal para Discursos de Javier Milei
===================================================

Este módulo contiene la implementación principal del sistema RAG que orquesta
todos los componentes: web scraping, procesamiento de documentos, embeddings,
búsqueda vectorial y generación de respuestas con LLM.

Autor: Sistema RAG Milei
Fecha: 2025-11-16
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

# Componentes del sistema
from .document_processor import DocumentProcessor
from .web_scraper import WebScraper
from .embeddings_handler import EmbeddingsHandler
from .vector_store import VectorStore
from .llm_handler import LLMHandler
from .utils import Utils

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Sistema RAG principal que coordina todos los componentes del pipeline.
    
    Encapsula la funcionalidad completa de:
    - Ingesta de documentos (web scraping + archivos locales)
    - Procesamiento y chunking inteligente
    - Generación de embeddings
    - Almacenamiento vectorial con ChromaDB
    - Recuperación semántica optimizada
    - Generación de respuestas con Gemini
    - Citación automática de fuentes
    """
    
    def __init__(self, 
                 corpus_path: str = "data/corpus",
                 vector_db_path: str = "data/vector_db",
                 embedding_model: str = "intfloat/multilingual-e5-large",
                 llm_model: str = "gemini-1.5-flash",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 top_k: int = 4):
        """
        Inicializar el sistema RAG.
        
        Args:
            corpus_path: Directorio para documentos corpus
            vector_db_path: Directorio para base de datos vectorial
            embedding_model: Modelo de embeddings a utilizar
            llm_model: Modelo LLM para generación de respuestas
            chunk_size: Tamaño de fragmentos de texto
            chunk_overlap: Superposición entre fragmentos
            top_k: Número de documentos a recuperar en búsquedas
        """
        # Configuración del sistema
        self.corpus_path = corpus_path
        self.vector_db_path = vector_db_path
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        # Crear directorios si no existen
        os.makedirs(corpus_path, exist_ok=True)
        os.makedirs(vector_db_path, exist_ok=True)
        
        # Inicializar componentes
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.web_scraper = WebScraper()
        
        self.embeddings_handler = EmbeddingsHandler(
            model_name=embedding_model
        )
        
        self.vector_store = VectorStore(
            persist_directory=vector_db_path
        )
        
        self.llm_handler = LLMHandler(
            model=llm_model
        )
        
        self.utils = Utils()
        
        # Estado del sistema
        self.is_initialized = False
        self.is_indexed = False
        self.document_count = 0
        
        logger.info(f"Sistema RAG inicializado con modelo embeddings: {embedding_model}")
    
    def initialize(self) -> Dict[str, Any]:
        """
        Inicializar todos los componentes del sistema.
        
        Returns:
            Dict con el estado de inicialización
        """
        try:
            logger.info("🚀 Inicializando sistema RAG...")
            
            # Verificar y crear base de datos vectorial
            self.vector_store.initialize()
            
            # Verificar si hay documentos indexados
            self.document_count = self.vector_store.get_document_count()
            self.is_indexed = self.document_count > 0
            
            self.is_initialized = True
            
            status = {
                "success": True,
                "system_ready": True,
                "documents_indexed": self.document_count,
                "vector_db_ready": True,
                "embeddings_model": self.embedding_model,
                "llm_model": self.llm_model,
                "chunk_size": self.chunk_size,
                "top_k": self.top_k,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Sistema RAG inicializado correctamente")
            logger.info(f"📊 Documentos indexados: {self.document_count}")
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Error inicializando sistema RAG: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def ingest_documents(self, 
                        source: str = "web",
                        max_documents: int = 10,
                        force_refresh: bool = False) -> Dict[str, Any]:
        """
        Ingesta y procesa documentos desde la fuente especificada.
        
        Args:
            source: Fuente de documentos ("web", "local", "both")
            max_documents: Máximo número de documentos a procesar
            force_refresh: Forzar re-procesamiento de documentos existentes
            
        Returns:
            Dict con resultados de la ingesta
        """
        if not self.is_initialized:
            return {"success": False, "error": "Sistema no inicializado"}
        
        try:
            logger.info(f"📥 Iniciando ingesta desde: {source}")
            
            start_time = datetime.now()
            documents_processed = 0
            documents_failed = 0
            
            # Web scraping
            if source in ["web", "both"]:
                try:
                    logger.info("🕷️ Ejecutando web scraping...")
                    scraped_docs = self.web_scraper.scrape_discursos(
                        max_discursos=max_documents
                    )
                    
                    if scraped_docs:
                        # Procesar documentos scrapeados
                        for doc in scraped_docs:
                            try:
                                # Guardar documento
                                filename = self._save_document(doc, "web")
                                
                                # Procesar y agregar al vector store
                                processed_chunks = self.document_processor.process_document(
                                    filename, 
                                    doc["metadata"]
                                )
                                
                                if processed_chunks:
                                    self.vector_store.add_documents(processed_chunks)
                                    documents_processed += 1
                                    
                            except Exception as e:
                                logger.warning(f"Error procesando documento web: {e}")
                                documents_failed += 1
                                
                except Exception as e:
                    logger.error(f"Error en web scraping: {e}")
            
            # Procesamiento de documentos locales
            if source in ["local", "both"]:
                try:
                    logger.info("📁 Procesando documentos locales...")
                    local_docs = self.utils.get_local_documents(self.corpus_path)
                    
                    for doc_path in local_docs[:max_documents]:
                        try:
                            processed_chunks = self.document_processor.process_document(
                                doc_path
                            )
                            
                            if processed_chunks:
                                self.vector_store.add_documents(processed_chunks)
                                documents_processed += 1
                                
                        except Exception as e:
                            logger.warning(f"Error procesando documento local: {e}")
                            documents_failed += 1
                            
                except Exception as e:
                    logger.error(f"Error procesando documentos locales: {e}")
            
            # Actualizar estado
            self.document_count = self.vector_store.get_document_count()
            self.is_indexed = self.document_count > 0
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            result = {
                "success": True,
                "source": source,
                "documents_processed": documents_processed,
                "documents_failed": documents_failed,
                "total_documents": self.document_count,
                "processing_time_seconds": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Ingesta completada: {documents_processed} documentos procesados")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en ingesta de documentos: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def query(self, 
              question: str, 
              include_sources: bool = True,
              max_response_length: int = 2000) -> Dict[str, Any]:
        """
        Procesar una consulta usando el sistema RAG completo.
        
        Args:
            question: Pregunta del usuario
            include_sources: Si incluir fuentes en la respuesta
            max_response_length: Longitud máxima de respuesta
            
        Returns:
            Dict con la respuesta y metadatos
        """
        if not self.is_initialized or not self.is_indexed:
            return {
                "success": False,
                "error": "Sistema no inicializado o sin documentos indexados",
                "question": question
            }
        
        try:
            start_time = datetime.now()
            
            # 1. Buscar documentos relevantes
            logger.info(f"🔍 Buscando documentos relevantes para: {question[:50]}...")
            relevant_docs = self.vector_store.similarity_search(
                query=question,
                k=self.top_k
            )
            
            if not relevant_docs:
                return {
                    "success": False,
                    "error": "No se encontraron documentos relevantes",
                    "question": question,
                    "sources": []
                }
            
            # 2. Generar respuesta con LLM
            logger.info("🤖 Generando respuesta con LLM...")
            response_data = self.llm_handler.generate_response(
                question=question,
                context_docs=relevant_docs,
                include_sources=include_sources,
                max_length=max_response_length
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # 3. Formatear resultado final
            result = {
                "success": True,
                "question": question,
                "answer": response_data["answer"],
                "sources": response_data["sources"] if include_sources else [],
                "confidence": response_data.get("confidence", 0.0),
                "processing_time_seconds": processing_time,
                "documents_retrieved": len(relevant_docs),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Consulta procesada en {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error procesando consulta: {e}")
            return {
                "success": False,
                "error": str(e),
                "question": question,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas completas del sistema.
        
        Returns:
            Dict con estadísticas del sistema
        """
        try:
            stats = {
                "system_info": {
                    "version": "1.0.0",
                    "initialized": self.is_initialized,
                    "indexed": self.is_indexed,
                    "corpus_path": self.corpus_path,
                    "vector_db_path": self.vector_db_path
                },
                "configuration": {
                    "embedding_model": self.embedding_model,
                    "llm_model": self.llm_model,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "top_k": self.top_k
                },
                "vector_store": self.vector_store.get_stats(),
                "timestamp": datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {"error": str(e)}
    
    def _save_document(self, doc_data: Dict, source: str) -> str:
        """
        Guardar documento procesado en el corpus.
        
        Args:
            doc_data: Datos del documento
            source: Fuente del documento
            
        Returns:
            Ruta del archivo guardado
        """
        try:
            # Crear nombre de archivo único
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = self.utils.sanitize_filename(doc_data.get("title", "documento"))
            filename = f"{timestamp}_{safe_title}_{source}.txt"
            filepath = os.path.join(self.corpus_path, filename)
            
            # Guardar contenido
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Título: {doc_data.get('title', 'Sin título')}\n")
                f.write(f"Fecha: {doc_data.get('fecha', 'Fecha desconocida')}\n")
                f.write(f"Fuente: {doc_data.get('url', 'URL desconocida')}\n")
                f.write(f"Tipo: {doc_data.get('tipo', 'Discurso')}\n")
                f.write("-" * 80 + "\n\n")
                f.write(doc_data.get("contenido", ""))
            
            logger.info(f"📄 Documento guardado: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error guardando documento: {e}")
            return ""
    
    def clear_database(self) -> Dict[str, Any]:
        """
        Limpiar completamente la base de datos vectorial.
        
        Returns:
            Dict con resultado de la operación
        """
        try:
            self.vector_store.clear()
            self.document_count = 0
            self.is_indexed = False
            
            logger.info("🗑️ Base de datos vectorial limpiada")
            
            return {
                "success": True,
                "message": "Base de datos limpiada correctamente",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error limpiando base de datos: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }