"""
Manejador de Embeddings para Sistema RAG
=======================================

Este módulo maneja la generación y gestión de embeddings de texto usando
modelos de Sentence Transformers optimizados para español.

Funcionalidades:
- Inicialización y configuración de modelos de embeddings
- Generación de embeddings para documentos y consultas
- Cache de embeddings para optimización de performance
- Soporte para modelos multilenguaje
- Batch processing para documentos grandes

Autor: Sistema RAG Milei
Fecha: 2025-11-16
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union
import numpy as np
from datetime import datetime
import hashlib
import pickle
from pathlib import Path

# ML y NLP
import torch
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Configuración de logging
logger = logging.getLogger(__name__)


class EmbeddingsHandler:
    """
    Manejador de embeddings optimizado para el sistema RAG.
    
    Características:
    - Modelos pre-entrenados optimizados para español
    - Cache de embeddings para evitar re-cálculo
    - Batch processing para performance
    - Soporte para múltiples modelos
    - Métricas de calidad de embeddings
    """
    
    def __init__(self, 
                 model_name: str = "intfloat/multilingual-e5-large",
                 cache_dir: str = "data/cache/embeddings",
                 batch_size: int = 32,
                 device: Optional[str] = None,
                 normalize_embeddings: bool = True):
        """
        Inicializar manejador de embeddings.
        
        Args:
            model_name: Nombre del modelo de Sentence Transformers
            cache_dir: Directorio para cache de embeddings
            batch_size: Tamaño de lote para procesamiento
            device: Dispositivo para cómputo ('cpu', 'cuda', None=auto)
            normalize_embeddings: Si normalizar embeddings a unit length
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        
        # Crear directorio de cache
        os.makedirs(cache_dir, exist_ok=True)
        
        # Determinar dispositivo
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Inicializar modelo
        self.model = None
        self.langchain_embeddings = None
        self.model_loaded = False
        
        # Cache en memoria para embeddings recientes
        self.memory_cache = {}
        self.max_memory_cache_size = 1000
        
        logger.info(f"🧠 EmbeddingsHandler inicializado para modelo: {model_name}")
        logger.info(f"💻 Usando dispositivo: {self.device}")
    
    def initialize(self) -> Dict[str, Any]:
        """
        Inicializar el modelo de embeddings.
        
        Returns:
            Dict con información del modelo cargado
        """
        try:
            logger.info(f"🚀 Cargando modelo de embeddings: {self.model_name}")
            
            # Cargar modelo Sentence Transformers
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Cargar wrapper de LangChain
            self.langchain_embeddings = SentenceTransformerEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': self.device}
            )
            
            self.model_loaded = True
            
            # Obtener información del modelo
            model_info = self._get_model_info()
            
            result = {
                "success": True,
                "model_name": self.model_name,
                "device": self.device,
                "model_info": model_info,
                "cache_dir": self.cache_dir,
                "batch_size": self.batch_size,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Modelo de embeddings cargado exitosamente")
            logger.info(f"📊 Dimensiones: {model_info.get('embedding_dimension', 'Unknown')}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo de embeddings: {e}")
            self.model_loaded = False
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generar embeddings para una lista de documentos.
        
        Args:
            texts: Lista de textos a procesar
            
        Returns:
            Lista de arrays numpy con embeddings
        """
        if not self.model_loaded:
            raise RuntimeError("Modelo no inicializado. Llama a initialize() primero.")
        
        if not texts:
            return []
        
        try:
            # Verificar cache para textos individuales
            cached_embeddings = []
            texts_to_process = []
            indices_to_process = []
            
            for i, text in enumerate(texts):
                cache_key = self._generate_cache_key(text, "document")
                cached_embedding = self._get_from_cache(cache_key)
                
                if cached_embedding is not None:
                    cached_embeddings.append((i, cached_embedding))
                else:
                    texts_to_process.append(text)
                    indices_to_process.append(i)
            
            # Procesar textos no cacheados
            if texts_to_process:
                logger.debug(f"Generando embeddings para {len(texts_to_process)} textos")
                
                # Batch processing
                processed_embeddings = []
                
                for i in range(0, len(texts_to_process), self.batch_size):
                    batch = texts_to_process[i:i + self.batch_size]
                    
                    try:
                        # Generar embeddings con SentenceTransformer
                        batch_embeddings = self.model.encode(
                            batch,
                            batch_size=len(batch),
                            show_progress_bar=False,
                            convert_to_numpy=True,
                            normalize_embeddings=self.normalize_embeddings
                        )
                        
                        if self.normalize_embeddings and len(batch_embeddings.shape) == 2:
                            # Normalizar si es necesario
                            batch_embeddings = self._normalize_embeddings(batch_embeddings)
                        
                        processed_embeddings.extend(batch_embeddings)
                        
                    except Exception as e:
                        logger.error(f"Error procesando batch {i//self.batch_size + 1}: {e}")
                        # En caso de error, usar embeddings zeros
                        zero_embeddings = [np.zeros(self.get_embedding_dimension()) for _ in batch]
                        processed_embeddings.extend(zero_embeddings)
                
                # Guardar en cache y mezclar con resultados cacheados
                final_embeddings = [None] * len(texts)
                
                # Colocar embeddings cacheados
                for idx, embedding in cached_embeddings:
                    final_embeddings[idx] = embedding
                
                # Colocar embeddings procesados
                for i, idx in enumerate(indices_to_process):
                    if i < len(processed_embeddings):
                        embedding = processed_embeddings[i]
                        final_embeddings[idx] = embedding
                        
                        # Guardar en cache
                        cache_key = self._generate_cache_key(texts[idx], "document")
                        self._save_to_cache(cache_key, embedding)
                
                return final_embeddings
            else:
                # Todos estaban cacheados
                result = [None] * len(texts)
                for idx, embedding in cached_embeddings:
                    result[idx] = embedding
                return result
                
        except Exception as e:
            logger.error(f"Error generando embeddings para documentos: {e}")
            # Retornar embeddings de zeros en caso de error
            return [np.zeros(self.get_embedding_dimension()) for _ in texts]
    
    def embed_query(self, text: str) -> np.ndarray:
        """
        Generar embedding para una consulta.
        
        Args:
            text: Texto de la consulta
            
        Returns:
            Array numpy con el embedding
        """
        if not self.model_loaded:
            raise RuntimeError("Modelo no inicializado. Llama a initialize() primero.")
        
        if not text.strip():
            return np.zeros(self.get_embedding_dimension())
        
        try:
            # Verificar cache
            cache_key = self._generate_cache_key(text, "query")
            cached_embedding = self._get_from_cache(cache_key)
            
            if cached_embedding is not None:
                return cached_embedding
            
            # Generar embedding
            embedding = self.model.encode(
                [text],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings
            )[0]
            
            # Normalizar si es necesario
            if self.normalize_embeddings:
                embedding = self._normalize_embeddings(embedding)
            
            # Guardar en cache
            self._save_to_cache(cache_key, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generando embedding para consulta: {e}")
            return np.zeros(self.get_embedding_dimension())
    
    def get_embedding_dimension(self) -> int:
        """Obtener la dimensión de los embeddings."""
        if not self.model_loaded:
            return 0
        
        try:
            return self.model.get_sentence_embedding_dimension()
        except Exception:
            return 768  # Default para modelos multilingües
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Obtener información detallada del modelo."""
        try:
            info = {
                "model_name": self.model_name,
                "device": self.device,
                "embedding_dimension": self.get_embedding_dimension(),
                "max_seq_length": getattr(self.model, 'max_seq_length', 'Unknown'),
                "normalize_embeddings": self.normalize_embeddings,
                "batch_size": self.batch_size
            }
            
            # Intentar obtener más información del modelo
            try:
                if hasattr(self.model, 'card'):
                    info["model_card"] = str(self.model.card)
            except Exception:
                pass
            
            # Información de configuración del modelo
            try:
                config = getattr(self.model, 'config', None)
                if config:
                    info["model_config"] = str(config)
            except Exception:
                pass
            
            return info
            
        except Exception as e:
            logger.error(f"Error obteniendo info del modelo: {e}")
            return {"error": str(e)}
    
    def _generate_cache_key(self, text: str, text_type: str) -> str:
        """Generar clave de cache para un texto."""
        # Hash del contenido del texto
        content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # Incluir información del modelo y tipo
        key_data = f"{self.model_name}:{text_type}:{content_hash}"
        cache_key = hashlib.md5(key_data.encode('utf-8')).hexdigest()
        
        return cache_key
    
    def _get_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Obtener embedding desde cache."""
        try:
            # Verificar cache en memoria primero
            if cache_key in self.memory_cache:
                return self.memory_cache[cache_key]
            
            # Verificar cache en disco
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    embedding = pickle.load(f)
                    
                    # Agregar a cache en memoria
                    self._add_to_memory_cache(cache_key, embedding)
                    
                    return embedding
            
            return None
            
        except Exception as e:
            logger.debug(f"Error obteniendo de cache: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, embedding: np.ndarray):
        """Guardar embedding en cache."""
        try:
            # Agregar a cache en memoria
            self._add_to_memory_cache(cache_key, embedding)
            
            # Guardar en disco
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
                
        except Exception as e:
            logger.debug(f"Error guardando en cache: {e}")
    
    def _add_to_memory_cache(self, cache_key: str, embedding: np.ndarray):
        """Agregar embedding al cache en memoria."""
        # Limpiar cache si está lleno
        if len(self.memory_cache) >= self.max_memory_cache_size:
            # Eliminar el 20% más antiguo
            keys_to_remove = list(self.memory_cache.keys())[:self.max_memory_cache_size // 5]
            for key in keys_to_remove:
                del self.memory_cache[key]
        
        self.memory_cache[cache_key] = embedding
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalizar embeddings a unit length."""
        try:
            if len(embeddings.shape) == 1:
                # Embedding único
                norm = np.linalg.norm(embeddings)
                if norm > 0:
                    return embeddings / norm
                else:
                    return embeddings
            else:
                # Batch de embeddings
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                # Evitar división por cero
                norms = np.where(norms == 0, 1, norms)
                return embeddings / norms
        except Exception as e:
            logger.debug(f"Error normalizando embeddings: {e}")
            return embeddings
    
    def clear_cache(self) -> Dict[str, Any]:
        """Limpiar todos los caches."""
        try:
            # Limpiar cache en memoria
            self.memory_cache.clear()
            
            # Limpiar cache en disco
            if os.path.exists(self.cache_dir):
                for file in os.listdir(self.cache_dir):
                    if file.endswith('.pkl'):
                        os.remove(os.path.join(self.cache_dir, file))
            
            logger.info("🧹 Cache de embeddings limpiado")
            
            return {
                "success": True,
                "message": "Cache limpiado correctamente",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error limpiando cache: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache."""
        try:
            cache_files = 0
            cache_size = 0
            
            if os.path.exists(self.cache_dir):
                for file in os.listdir(self.cache_dir):
                    if file.endswith('.pkl'):
                        cache_files += 1
                        cache_size += os.path.getsize(os.path.join(self.cache_dir, file))
            
            return {
                "memory_cache_size": len(self.memory_cache),
                "disk_cache_files": cache_files,
                "disk_cache_size_bytes": cache_size,
                "disk_cache_size_mb": round(cache_size / (1024 * 1024), 2),
                "cache_dir": self.cache_dir,
                "model_name": self.model_name,
                "device": self.device
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo stats del cache: {e}")
            return {"error": str(e)}
    
    def benchmark_model(self, test_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """Benchmark del modelo de embeddings."""
        if not test_texts:
            test_texts = [
                "El presidente Milei habló sobre la economía argentina",
                "La inflación es un problema grave que debemos resolver",
                "El modelo económico necesita cambios profundos",
                "La libertad económica es fundamental para el crecimiento"
            ]
        
        try:
            import time
            
            logger.info("🏃 Iniciando benchmark del modelo de embeddings")
            
            # Benchmark de velocidad
            start_time = time.time()
            embeddings = self.embed_documents(test_texts)
            end_time = time.time()
            
            processing_time = end_time - start_time
            avg_time_per_text = processing_time / len(test_texts)
            
            # Calcular estadísticas de embeddings
            embedding_dim = self.get_embedding_dimension()
            avg_norm = np.mean([np.linalg.norm(emb) for emb in embeddings])
            
            # Calcular similitud coseno entre embeddings
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    similarities.append(sim)
            
            avg_similarity = np.mean(similarities) if similarities else 0
            
            result = {
                "success": True,
                "model_name": self.model_name,
                "test_texts_count": len(test_texts),
                "total_processing_time_seconds": processing_time,
                "avg_time_per_text_seconds": avg_time_per_text,
                "texts_per_second": len(test_texts) / processing_time if processing_time > 0 else 0,
                "embedding_dimension": embedding_dim,
                "avg_embedding_norm": avg_norm,
                "avg_cosine_similarity": avg_similarity,
                "device": self.device,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Benchmark completado: {len(test_texts)} textos en {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error en benchmark: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }