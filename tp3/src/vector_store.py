"""
Vector Store para Sistema RAG - ChromaDB Integration
=================================================

Este módulo implementa el manejo de la base de datos vectorial usando ChromaDB
para el almacenamiento y búsqueda eficiente de embeddings.

Funcionalidades:
- Inicialización y configuración de ChromaDB
- Almacenamiento persistente de documentos y embeddings
- Búsqueda semántica con similarity search
- Filtrado por metadata
- Estrategias de búsqueda avanzadas (MMR, threshold)
- Métricas y estadísticas de la base de datos

Autor: Sistema RAG Milei
Fecha: 2025-11-16
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import json
import sqlite3
from pathlib import Path

# ChromaDB
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# LangChain integration
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Documentos
from langchain.schema import Document as LangChainDocument

# Configuración de logging
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manejador de la base de datos vectorial ChromaDB.
    
    Funcionalidades:
    - Almacenamiento de documentos con embeddings
    - Búsqueda semántica optimizada
    - Persistencia en disco
    - Filtros por metadata
    - Estadísticas y métricas
    - Estrategias de búsqueda avanzadas
    """
    
    def __init__(self, 
                 persist_directory: str = "data/vector_db",
                 collection_name: str = "milei_discursos",
                 embedding_function_name: str = "intfloat/multilingual-e5-large",
                 chroma_server_host: Optional[str] = None,
                 chroma_server_port: Optional[int] = None):
        """
        Inicializar vector store.
        
        Args:
            persist_directory: Directorio para persistencia local
            collection_name: Nombre de la colección
            embedding_function_name: Nombre del modelo de embeddings
            chroma_server_host: Host del servidor ChromaDB (opcional)
            chroma_server_port: Puerto del servidor ChromaDB (opcional)
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_function_name = embedding_function_name
        
        # Crear directorio si no existe
        os.makedirs(persist_directory, exist_ok=True)
        
        # Configurar cliente ChromaDB
        self.client = None
        self.collection = None
        self.langchain_vectorstore = None
        
        # Configuración de ChromaDB
        self.chroma_settings = Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False,
            allow_reset=True
        )
        
        # Estado del sistema
        self.is_initialized = False
        self.document_count = 0
        self.last_updated = None
        
        logger.info(f"🗄️ VectorStore inicializado: {collection_name}")
    
    def initialize(self) -> Dict[str, Any]:
        """
        Inicializar la base de datos vectorial.
        
        Returns:
            Dict con estado de inicialización
        """
        try:
            logger.info("🚀 Inicializando ChromaDB...")
            
            # Configurar función de embeddings
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_function_name
            )
            
            # Crear cliente ChromaDB
            if self.chroma_settings.persist_directory:
                self.client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=self.chroma_settings
                )
            else:
                self.client = chromadb.HttpClient(
                    host=self.chroma_server_host or "localhost",
                    port=self.chroma_server_port or 8000
                )
            
            # Obtener o crear colección
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"📁 Colección existente encontrada: {self.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"description": "Discursos de Javier Milei - Sistema RAG"}
                )
                logger.info(f"🆕 Nueva colección creada: {self.collection_name}")
            
            # Crear wrapper de LangChain
            self.langchain_vectorstore = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory if self.chroma_settings.persist_directory else None
            )
            
            # Actualizar estadísticas
            self._update_stats()
            self.is_initialized = True
            
            result = {
                "success": True,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "embedding_function": self.embedding_function_name,
                "document_count": self.document_count,
                "chroma_version": chromadb.__version__,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ ChromaDB inicializado correctamente")
            logger.info(f"📊 Documentos en base: {self.document_count}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error inicializando ChromaDB: {e}")
            self.is_initialized = False
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def add_documents(self, documents: List[LangChainDocument]) -> Dict[str, Any]:
        """
        Agregar documentos a la base de datos vectorial.
        
        Args:
            documents: Lista de documentos LangChain
            
        Returns:
            Dict con resultado de la operación
        """
        if not self.is_initialized:
            return {"success": False, "error": "VectorStore no inicializado"}
        
        if not documents:
            return {"success": True, "message": "No hay documentos para agregar"}
        
        try:
            logger.info(f"📝 Agregando {len(documents)} documentos a la base vectorial")
            
            # Preparar datos para ChromaDB
            ids = []
            documents_text = []
            metadatas = []
            
            for i, doc in enumerate(documents):
                doc_id = doc.metadata.get('chunk_id', i + 1)
                # Crear ID único
                unique_id = f"doc_{doc_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                ids.append(unique_id)
                documents_text.append(doc.page_content)
                metadatas.append(doc.metadata)
            
            # Agregar a ChromaDB
            self.collection.add(
                ids=ids,
                documents=documents_text,
                metadatas=metadatas
            )
            
            # Actualizar estadísticas
            self._update_stats()
            
            result = {
                "success": True,
                "documents_added": len(documents),
                "total_documents": self.document_count,
                "collection_name": self.collection_name,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ {len(documents)} documentos agregados exitosamente")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error agregando documentos: {e}")
            return {
                "success": False,
                "error": str(e),
                "documents_attempted": len(documents),
                "timestamp": datetime.now().isoformat()
            }
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 4, 
                         filter_dict: Optional[Dict] = None,
                         score_threshold: Optional[float] = None,
                         search_type: str = "similarity") -> List[LangChainDocument]:
        """
        Realizar búsqueda semántica en la base de datos.
        
        Args:
            query: Consulta de búsqueda
            k: Número de resultados a retornar
            filter_dict: Filtros por metadata
            score_threshold: Umbral de score mínimo
            search_type: Tipo de búsqueda ("similarity", "mmr", "similarity_score_threshold")
            
        Returns:
            Lista de documentos LangChain ordenados por relevancia
        """
        if not self.is_initialized:
            raise RuntimeError("VectorStore no inicializado")
        
        try:
            logger.debug(f"🔍 Búsqueda: '{query[:50]}...' (k={k}, type={search_type})")
            
            # Preparar parámetros de búsqueda
            search_kwargs = {"k": k}
            
            if filter_dict:
                search_kwargs["filter"] = filter_dict
            
            if search_type == "similarity_score_threshold" and score_threshold:
                search_kwargs["score_threshold"] = score_threshold
            
            # Realizar búsqueda usando LangChain wrapper
            if search_type == "mmr":
                # MMR search requiere parámetros especiales
                search_kwargs.update({
                    "fetch_k": k * 2,  # Buscar más documentos para diversificar
                    "lambda_mult": 0.7  # Balance entre relevancia y diversidad
                })
            
            # Usar el retriever de LangChain
            retriever = self.langchain_vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs
            )
            
            # Realizar búsqueda
            results = retriever.get_relevant_documents(query)
            
            logger.debug(f"✅ Búsqueda completada: {len(results)} resultados")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en búsqueda semántica: {e}")
            return []
    
    def get_document_by_id(self, doc_id: str) -> Optional[LangChainDocument]:
        """
        Obtener un documento específico por ID.
        
        Args:
            doc_id: ID del documento
            
        Returns:
            Documento LangChain o None si no se encuentra
        """
        if not self.is_initialized:
            return None
        
        try:
            # Buscar en ChromaDB
            results = self.collection.get(
                ids=[doc_id],
                include=["documents", "metadatas"]
            )
            
            if results["ids"] and len(results["ids"]) > 0:
                # Reconstruir documento LangChain
                doc = LangChainDocument(
                    page_content=results["documents"][0],
                    metadata=results["metadatas"][0]
                )
                return doc
            
            return None
            
        except Exception as e:
            logger.error(f"Error obteniendo documento {doc_id}: {e}")
            return None
    
    def update_document(self, doc_id: str, document: LangChainDocument) -> Dict[str, Any]:
        """
        Actualizar un documento existente.
        
        Args:
            doc_id: ID del documento a actualizar
            document: Nuevo contenido del documento
            
        Returns:
            Dict con resultado de la operación
        """
        if not self.is_initialized:
            return {"success": False, "error": "VectorStore no inicializado"}
        
        try:
            # Actualizar en ChromaDB
            self.collection.update(
                ids=[doc_id],
                documents=[document.page_content],
                metadatas=[document.metadata]
            )
            
            # Actualizar estadísticas
            self._update_stats()
            
            logger.info(f"📝 Documento {doc_id} actualizado")
            
            return {
                "success": True,
                "doc_id": doc_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error actualizando documento {doc_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "doc_id": doc_id,
                "timestamp": datetime.now().isoformat()
            }
    
    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Eliminar un documento de la base de datos.
        
        Args:
            doc_id: ID del documento a eliminar
            
        Returns:
            Dict con resultado de la operación
        """
        if not self.is_initialized:
            return {"success": False, "error": "VectorStore no inicializado"}
        
        try:
            # Eliminar de ChromaDB
            self.collection.delete(ids=[doc_id])
            
            # Actualizar estadísticas
            self._update_stats()
            
            logger.info(f"🗑️ Documento {doc_id} eliminado")
            
            return {
                "success": True,
                "doc_id": doc_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error eliminando documento {doc_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "doc_id": doc_id,
                "timestamp": datetime.now().isoformat()
            }
    
    def clear(self) -> Dict[str, Any]:
        """
        Limpiar completamente la base de datos.
        
        Returns:
            Dict con resultado de la operación
        """
        if not self.is_initialized:
            return {"success": False, "error": "VectorStore no inicializado"}
        
        try:
            # Eliminar todos los documentos
            self.collection.delete(where={})
            
            # Resetear contador
            self.document_count = 0
            self.last_updated = datetime.now()
            
            logger.info("🧹 Base de datos vectorial limpiada")
            
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
    
    def get_document_count(self) -> int:
        """Obtener número total de documentos."""
        if not self.is_initialized:
            return 0
        
        try:
            count = self.collection.count()
            self.document_count = count
            return count
        except Exception as e:
            logger.error(f"Error obteniendo conteo de documentos: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas detalladas de la base de datos.
        
        Returns:
            Dict con estadísticas completas
        """
        if not self.is_initialized:
            return {"error": "VectorStore no inicializado"}
        
        try:
            # Información básica
            stats = {
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "embedding_function": self.embedding_function_name,
                "document_count": self.document_count,
                "last_updated": self.last_updated.isoformat() if self.last_updated else None,
                "chroma_version": chromadb.__version__,
                "is_initialized": self.is_initialized
            }
            
            # Información de la colección
            collection_info = self.collection.count()
            stats["actual_count"] = collection_info
            
            # Obtener sample de metadatos para análisis
            try:
                sample_results = self.collection.get(limit=10)
                if sample_results["metadatas"]:
                    metadata_keys = set()
                    for metadata in sample_results["metadatas"]:
                        metadata_keys.update(metadata.keys())
                    stats["metadata_fields"] = list(metadata_keys)
                    
                    # Estadísticas de tipos de documento
                    doc_types = {}
                    for metadata in sample_results["metadatas"]:
                        doc_type = metadata.get("doc_type", "unknown")
                        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                    stats["document_types"] = doc_types
                    
            except Exception:
                pass
            
            # Información de almacenamiento
            if os.path.exists(self.persist_directory):
                total_size = 0
                file_count = 0
                for root, dirs, files in os.walk(self.persist_directory):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            total_size += os.path.getsize(file_path)
                            file_count += 1
                
                stats["storage_size_bytes"] = total_size
                stats["storage_size_mb"] = round(total_size / (1024 * 1024), 2)
                stats["storage_files"] = file_count
            
            return stats
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {"error": str(e)}
    
    def list_documents(self, 
                      limit: int = 50, 
                      offset: int = 0, 
                      filter_dict: Optional[Dict] = None) -> List[LangChainDocument]:
        """
        Listar documentos con paginación y filtros.
        
        Args:
            limit: Número máximo de documentos
            offset: Offset para paginación
            filter_dict: Filtros por metadata
            
        Returns:
            Lista de documentos
        """
        if not self.is_initialized:
            return []
        
        try:
            # Parámetros para consulta
            query_params = {
                "limit": limit,
                "offset": offset,
                "include": ["documents", "metadatas"]
            }
            
            if filter_dict:
                query_params["where"] = filter_dict
            
            # Obtener documentos
            results = self.collection.get(**query_params)
            
            # Convertir a documentos LangChain
            documents = []
            if results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    doc = LangChainDocument(
                        page_content=results["documents"][i],
                        metadata=results["metadatas"][i]
                    )
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error listando documentos: {e}")
            return []
    
    def _update_stats(self):
        """Actualizar estadísticas internas."""
        try:
            self.document_count = self.collection.count()
            self.last_updated = datetime.now()
        except Exception as e:
            logger.error(f"Error actualizando estadísticas: {e}")
    
    def backup_database(self, backup_path: str) -> Dict[str, Any]:
        """
        Crear backup de la base de datos vectorial.
        
        Args:
            backup_path: Ruta donde guardar el backup
            
        Returns:
            Dict con resultado de la operación
        """
        try:
            import shutil
            
            if not os.path.exists(self.persist_directory):
                return {"success": False, "error": "Base de datos no encontrada"}
            
            # Crear directorio de backup si no existe
            backup_dir = os.path.dirname(backup_path)
            os.makedirs(backup_dir, exist_ok=True)
            
            # Copiar directorio completo
            shutil.copytree(self.persist_directory, backup_path, dirs_exist_ok=True)
            
            logger.info(f"💾 Backup creado en: {backup_path}")
            
            return {
                "success": True,
                "backup_path": backup_path,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creando backup: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }