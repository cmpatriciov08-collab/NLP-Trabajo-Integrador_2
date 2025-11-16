"""
LLM Handler para Sistema RAG - Google Gemini Integration
======================================================

Este módulo maneja la integración con Google Gemini para la generación
de respuestas contextuales usando el sistema RAG.

Funcionalidades:
- Configuración y manejo de Google Gemini
- Generación de respuestas con contexto recuperado
- Templates de prompts optimizados para RAG
- Citación automática de fuentes
- Manejo de errores y rate limiting
- Cache de respuestas para optimización
- Métricas de calidad de respuestas

Autor: Sistema RAG Milei
Fecha: 2025-11-16
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import hashlib
import json
from pathlib import Path

# LLM Integration
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.base import Chain

# Documentos
from langchain.schema import Document as LangChainDocument

# Configuración de logging
logger = logging.getLogger(__name__)


class LLMHandler:
    """
    Manejador de LLM para el sistema RAG usando Google Gemini.
    
    Funcionalidades:
    - Integración con Google Gemini 1.5 Flash
    - Generación de respuestas contextuales
    - Templates de prompts optimizados
    - Sistema de citación de fuentes
    - Cache de respuestas
    - Rate limiting y manejo de errores
    - Métricas de calidad
    """
    
    def __init__(self, 
                 model: str = "gemini-1.5-flash",
                 temperature: float = 0.1,
                 max_output_tokens: int = 2000,
                 top_p: float = 0.8,
                 top_k: int = 40,
                 cache_dir: str = "data/cache/responses",
                 enable_caching: bool = True):
        """
        Inicializar LLM Handler.
        
        Args:
            model: Modelo de Gemini a utilizar
            temperature: Temperatura para generación (0.0-1.0)
            max_output_tokens: Máximo número de tokens de salida
            top_p: Top-p sampling
            top_k: Top-k sampling
            cache_dir: Directorio para cache de respuestas
            enable_caching: Si habilitar cache de respuestas
        """
        self.model_name = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.cache_dir = cache_dir
        self.enable_caching = enable_caching
        
        # Crear directorio de cache
        if enable_caching:
            os.makedirs(cache_dir, exist_ok=True)
        
        # Configurar Gemini
        self.llm = None
        self.llm_initialized = False
        
        # Configuración de seguridad y rate limiting
        self.rate_limit_calls = 10  # llamadas por minuto
        self.rate_limit_window = 60  # segundos
        self.call_history = []
        
        # Estadísticas
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        
        logger.info(f"🤖 LLMHandler inicializado para modelo: {model}")
    
    def initialize(self, api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Inicializar el cliente de Gemini.
        
        Args:
            api_key: API key de Google (opcional, usa env var si no se proporciona)
            
        Returns:
            Dict con estado de inicialización
        """
        try:
            logger.info("🚀 Inicializando Google Gemini...")
            
            # Configurar API key
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
            elif not os.getenv("GOOGLE_API_KEY"):
                return {
                    "success": False,
                    "error": "API key de Google no encontrada. Configura GOOGLE_API_KEY en variables de entorno.",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Configurar Gemini
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            
            # Crear modelo LangChain
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                top_p=self.top_p,
                top_k=self.top_k,
                convert_system_message_to_human=True
            )
            
            self.llm_initialized = True
            
            # Obtener información del modelo
            model_info = self._get_model_info()
            
            result = {
                "success": True,
                "model_name": self.model_name,
                "configuration": model_info,
                "temperature": self.temperature,
                "max_tokens": self.max_output_tokens,
                "caching_enabled": self.enable_caching,
                "rate_limit": {
                    "calls_per_minute": self.rate_limit_calls,
                    "window_seconds": self.rate_limit_window
                },
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("✅ Google Gemini inicializado correctamente")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error inicializando Gemini: {e}")
            self.llm_initialized = False
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def generate_response(self, 
                         question: str, 
                         context_docs: List[LangChainDocument],
                         include_sources: bool = True,
                         max_length: int = 2000) -> Dict[str, Any]:
        """
        Generar respuesta usando el contexto recuperado.
        
        Args:
            question: Pregunta del usuario
            context_docs: Documentos de contexto recuperado
            include_sources: Si incluir fuentes en la respuesta
            max_length: Longitud máxima de respuesta
            
        Returns:
            Dict con respuesta y metadatos
        """
        if not self.llm_initialized:
            raise RuntimeError("LLM no inicializado. Llama a initialize() primero.")
        
        try:
            # Verificar rate limiting
            if not self._check_rate_limit():
                return {
                    "success": False,
                    "error": "Rate limit excedido. Intenta más tarde.",
                    "question": question
                }
            
            # Verificar cache
            if self.enable_caching:
                cache_key = self._generate_cache_key(question, context_docs)
                cached_response = self._get_from_cache(cache_key)
                if cached_response:
                    logger.info("📦 Respuesta obtenida desde cache")
                    return cached_response
            
            # Preparar contexto
            context_text = self._prepare_context(context_docs)
            
            # Crear prompt
            prompt = self._create_rag_prompt(question, context_text, include_sources)
            
            # Generar respuesta
            start_time = datetime.now()
            
            response = self.llm.invoke(prompt)
            end_time = datetime.now()
            
            processing_time = (end_time - start_time).total_seconds()
            
            # Procesar respuesta
            answer_text = response.content if hasattr(response, 'content') else str(response)
            
            # Preparar fuentes
            sources = []
            if include_sources:
                sources = self._prepare_sources(context_docs)
            
            # Crear resultado
            result = {
                "success": True,
                "question": question,
                "answer": answer_text[:max_length],
                "sources": sources,
                "confidence": self._calculate_confidence(context_docs),
                "processing_time_seconds": processing_time,
                "model_used": self.model_name,
                "context_documents_count": len(context_docs),
                "timestamp": datetime.now().isoformat()
            }
            
            # Guardar en cache
            if self.enable_caching:
                self._save_to_cache(cache_key, result)
            
            # Actualizar estadísticas
            self._update_call_stats(True)
            
            logger.info(f"✅ Respuesta generada en {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error generando respuesta: {e}")
            self._update_call_stats(False)
            return {
                "success": False,
                "error": str(e),
                "question": question,
                "timestamp": datetime.now().isoformat()
            }
    
    def _prepare_context(self, context_docs: List[LangChainDocument]) -> str:
        """
        Preparar texto de contexto desde documentos recuperados.
        
        Args:
            context_docs: Lista de documentos de contexto
            
        Returns:
            Texto de contexto formateado
        """
        if not context_docs:
            return "No hay contexto disponible."
        
        context_parts = []
        
        for i, doc in enumerate(context_docs, 1):
            # Crear encabezado del documento
            source_info = doc.metadata.get('source', 'Fuente desconocida')
            title = doc.metadata.get('title', 'Sin título')
            
            # Agregar metadatos relevantes
            metadata_info = []
            if 'fecha' in doc.metadata:
                metadata_info.append(f"Fecha: {doc.metadata['fecha']}")
            if 'doc_type' in doc.metadata:
                metadata_info.append(f"Tipo: {doc.metadata['doc_type']}")
            
            metadata_str = " | ".join(metadata_info) if metadata_info else ""
            
            # Formatear documento
            doc_header = f"**DOCUMENTO {i}:**\n"
            doc_header += f"Fuente: {source_info}\n"
            if title != 'Sin título':
                doc_header += f"Título: {title}\n"
            if metadata_str:
                doc_header += f"{metadata_str}\n"
            doc_header += "\n"
            
            context_parts.append(doc_header + doc.page_content)
        
        return "\n\n" + "="*80 + "\n\n".join(context_parts)
    
    def _create_rag_prompt(self, question: str, context: str, include_sources: bool) -> str:
        """
        Crear prompt optimizado para RAG.
        
        Args:
            question: Pregunta del usuario
            context: Texto de contexto
            include_sources: Si incluir instrucciones sobre fuentes
            
        Returns:
            Prompt completo para el LLM
        """
        # Template base del prompt
        template = """Eres un asistente experto en análisis de discursos y políticas públicas argentinas.

Tu trabajo es responder preguntas basándote EXCLUSIVAMENTE en la información proporcionada en el contexto de documentos.

INSTRUCCIONES IMPORTANTES:
1. Solo usa información que aparece explícitamente en el contexto
2. Si la información no es suficiente, dilo claramente
3. Sé preciso con nombres, fechas y datos específicos
4. Usa un tono profesional e informativo
5. Si hay información contradictoria en el contexto, señálalo

{include_sources}

CONTEXTO DE DOCUMENTOS:
{context}

PREGUNTA DEL USUARIO: {question}

INSTRUCCIONES PARA LA RESPUESTA:
- Responde de forma clara y estructurada
- Si es relevante, incluye fechas y fuentes específicas
- Si la pregunta no puede responderse con el contexto, dilo explícitamente
- Mantén la respuesta concisa pero informativa

RESPUESTA:"""

        # Agregar instrucciones sobre fuentes si se solicita
        sources_instruction = ""
        if include_sources:
            sources_instruction = """6. Al final de tu respuesta, incluye una sección "FUENTES CONSULTADAS:" que liste cada documento usado con su fuente y título"""

        # Completar template
        prompt = template.format(
            include_sources=sources_instruction,
            context=context,
            question=question
        )
        
        return prompt
    
    def _prepare_sources(self, context_docs: List[LangChainDocument]) -> List[Dict[str, str]]:
        """
        Preparar información de fuentes para citación.
        
        Args:
            context_docs: Lista de documentos de contexto
            
        Returns:
            Lista de diccionarios con información de fuentes
        """
        sources = []
        
        for i, doc in enumerate(context_docs, 1):
            source_info = {
                "document_id": i,
                "source": doc.metadata.get('source', 'Fuente desconocida'),
                "title": doc.metadata.get('title', 'Sin título'),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            }
            
            # Agregar información adicional si está disponible
            if 'fecha' in doc.metadata:
                source_info['date'] = doc.metadata['fecha']
            if 'url' in doc.metadata:
                source_info['url'] = doc.metadata['url']
            if 'doc_type' in doc.metadata:
                source_info['document_type'] = doc.metadata['doc_type']
            
            sources.append(source_info)
        
        return sources
    
    def _calculate_confidence(self, context_docs: List[LangChainDocument]) -> float:
        """
        Calcular nivel de confianza basado en el contexto.
        
        Args:
            context_docs: Documentos de contexto
            
        Returns:
            Score de confianza entre 0.0 y 1.0
        """
        if not context_docs:
            return 0.0
        
        # Factores que aumentan la confianza
        confidence_factors = {
            "document_count": min(len(context_docs) / 5.0, 1.0),  # Más documentos = más confianza
            "content_quality": 0.0,  # A calcular basado en calidad del contenido
            "metadata_completeness": 0.0  # A calcular basado en metadata disponible
        }
        
        # Evaluar calidad del contenido
        total_content_length = sum(len(doc.page_content) for doc in context_docs)
        avg_content_length = total_content_length / len(context_docs)
        confidence_factors["content_quality"] = min(avg_content_length / 1000.0, 1.0)  # Normalizar por 1000 chars
        
        # Evaluar completitud de metadata
        metadata_completeness_scores = []
        for doc in context_docs:
            completeness = 0.0
            total_fields = 5  # fuente, título, fecha, tipo, url
            available_fields = 0
            
            for field in ['source', 'title', 'fecha', 'doc_type', 'url']:
                if field in doc.metadata and doc.metadata[field]:
                    available_fields += 1
            
            completeness = available_fields / total_fields
            metadata_completeness_scores.append(completeness)
        
        confidence_factors["metadata_completeness"] = sum(metadata_completeness_scores) / len(metadata_completeness_scores)
        
        # Calcular confianza final (promedio ponderado)
        final_confidence = (
            confidence_factors["document_count"] * 0.3 +
            confidence_factors["content_quality"] * 0.4 +
            confidence_factors["metadata_completeness"] * 0.3
        )
        
        return min(max(final_confidence, 0.0), 1.0)
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo Gemini."""
        try:
            # Obtener modelos disponibles
            models = [m.name for m in genai.list_models()]
            
            # Información específica del modelo actual
            model_info = {
                "model_name": self.model_name,
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "available_models": models,
                "current_model_available": f"models/{self.model_name}" in models
            }
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error obteniendo info del modelo: {e}")
            return {"error": str(e)}
    
    def _generate_cache_key(self, question: str, context_docs: List[LangChainDocument]) -> str:
        """Generar clave de cache para una consulta."""
        # Crear hash del contenido
        content_parts = [question]
        
        # Agregar hash de cada documento de contexto
        for doc in context_docs:
            doc_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()[:16]
            content_parts.append(doc_hash)
        
        # Hash final
        key_data = "|".join(content_parts)
        cache_key = hashlib.md5(key_data.encode('utf-8')).hexdigest()
        
        return cache_key
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Obtener respuesta desde cache."""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_response = json.load(f)
                    
                    # Verificar que no esté muy viejo (24 horas)
                    cached_time = datetime.fromisoformat(cached_response['timestamp'])
                    age_hours = (datetime.now() - cached_time).total_seconds() / 3600
                    
                    if age_hours < 24:  # Cache válido por 24 horas
                        return cached_response
                    else:
                        # Eliminar cache viejo
                        os.remove(cache_file)
            
            return None
            
        except Exception as e:
            logger.debug(f"Error obteniendo de cache: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, response: Dict[str, Any]):
        """Guardar respuesta en cache."""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(response, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug(f"Error guardando en cache: {e}")
    
    def _check_rate_limit(self) -> bool:
        """Verificar si se puede realizar otra llamada (rate limiting)."""
        try:
            now = datetime.now()
            
            # Limpiar historial viejo
            self.call_history = [
                call_time for call_time in self.call_history 
                if (now - call_time).total_seconds() < self.rate_limit_window
            ]
            
            # Verificar límite
            if len(self.call_history) >= self.rate_limit_calls:
                return False
            
            # Registrar esta llamada
            self.call_history.append(now)
            return True
            
        except Exception as e:
            logger.debug(f"Error verificando rate limit: {e}")
            return True  # En caso de error, permitir llamada
    
    def _update_call_stats(self, success: bool):
        """Actualizar estadísticas de llamadas."""
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
    
    def clear_cache(self) -> Dict[str, Any]:
        """Limpiar cache de respuestas."""
        if not self.enable_caching:
            return {"success": False, "error": "Cache deshabilitado"}
        
        try:
            if os.path.exists(self.cache_dir):
                for file in os.listdir(self.cache_dir):
                    if file.endswith('.json'):
                        os.remove(os.path.join(self.cache_dir, file))
            
            logger.info("🧹 Cache de respuestas limpiado")
            
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del LLM handler."""
        return {
            "model_name": self.model_name,
            "initialized": self.llm_initialized,
            "configuration": {
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "top_p": self.top_p,
                "top_k": self.top_k
            },
            "statistics": {
                "total_calls": self.total_calls,
                "successful_calls": self.successful_calls,
                "failed_calls": self.failed_calls,
                "success_rate": self.successful_calls / max(self.total_calls, 1),
                "current_calls_in_window": len(self.call_history)
            },
            "caching": {
                "enabled": self.enable_caching,
                "cache_dir": self.cache_dir,
                "rate_limit": {
                    "calls_per_minute": self.rate_limit_calls,
                    "window_seconds": self.rate_limit_window
                }
            }
        }