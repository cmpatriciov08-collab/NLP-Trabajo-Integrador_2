"""
Aplicación Streamlit - Sistema RAG Discursos de Javier Milei
============================================================

Interfaz web completa para el sistema RAG que permite:
- Chat conversacional con IA sobre discursos de Milei
- Administración de documentos (scraping, subida, procesamiento)
- Métricas y analytics del sistema
- Configuración de usuario y settings
- Historial de conversaciones
- Análisis de performance

Características:
- Diseño responsivo y profesional
- Gestión de estado con Streamlit sessions
- Integración completa con el sistema RAG
- Manejo de errores robusto
- Performance optimizada

Autor: Sistema RAG Milei  
Fecha: 2025-11-16
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Streamlit
import streamlit as st
import streamlit_option_menu
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Agregar src al path
sys.path.append(str(Path(__file__).parent / 'src'))

# Importar componentes del sistema RAG
from src.rag_system import RAGSystem
from src.utils import Utils

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la página
st.set_page_config(
    page_title="Sistema RAG - Discursos de Javier Milei",
    page_icon="🇦🇷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la apariencia
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        background-color: #f8f9fa;
    }
    
    .user-message {
        border-left-color: #28a745;
        background-color: #d4edda;
    }
    
    .assistant-message {
        border-left-color: #007bff;
        background-color: #d1ecf1;
    }
    
    .system-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    
    .status-success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)


class RAGApp:
    """
    Clase principal de la aplicación Streamlit.
    
    Maneja toda la lógica de la interfaz de usuario y la integración
    con el sistema RAG backend.
    """
    
    def __init__(self):
        """Inicializar la aplicación."""
        self.rag_system = None
        self.utils = Utils()
        self.init_session_state()
        
    def init_session_state(self):
        """Inicializar estado de sesión de Streamlit."""
        if 'rag_system_initialized' not in st.session_state:
            st.session_state.rag_system_initialized = False
        
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        
        if 'system_stats' not in st.session_state:
            st.session_state.system_stats = {}
        
        if 'user_settings' not in st.session_state:
            st.session_state.user_settings = {
                'auto_clear_history': False,
                'show_sources': True,
                'enable_caching': True,
                'response_style': 'detailed',
                'max_history_items': 50
            }
        
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
    
    def initialize_rag_system(self):
        """Inicializar el sistema RAG."""
        if not st.session_state.rag_system_initialized:
            try:
                with st.spinner("🚀 Inicializando Sistema RAG..."):
                    self.rag_system = RAGSystem()
                    
                    # Inicializar componentes
                    init_status = self.rag_system.initialize()
                    
                    if init_status.get('success'):
                        st.session_state.rag_system_initialized = True
                        st.session_state.system_stats = init_status
                        st.success("✅ Sistema RAG inicializado correctamente")
                        return True
                    else:
                        st.error(f"❌ Error inicializando sistema: {init_status.get('error')}")
                        return False
                        
            except Exception as e:
                st.error(f"❌ Error inesperado: {e}")
                logger.error(f"Error inicializando RAG: {e}")
                return False
        
        return True
    
    def render_header(self):
        """Renderizar encabezado principal."""
        st.markdown('<h1 class="main-header">🇦🇷 Sistema RAG - Discursos de Javier Milei</h1>', 
                   unsafe_allow_html=True)
        
        # Subtítulo y descripción
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            **Sistema de Recuperación y Generación Aumentada para Análisis de Discursos Presidenciales**
            
            💬 **Consulta inteligente** | 📊 **Análisis profundo** | 🎯 **Respuestas precisas**
            """)
        
        # Status del sistema
        if st.session_state.rag_system_initialized:
            st.markdown('<div class="system-status status-success">🟢 Sistema RAG - ACTIVO</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="system-status status-warning">🟡 Sistema RAG - INACTIVO</div>', 
                       unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Renderizar barra lateral con navegación."""
        with st.sidebar:
            st.markdown("### 🧭 Navegación")
            
            # Menú principal
            selected = option_menu(
                menu_title="Menú Principal",
                options=["💬 Chat RAG", "📁 Documentos", "📊 Analytics", "⚙️ Configuración"],
                icons=["chat-dots", "folder", "graph-up", "gear"],
                menu_icon="cast",
                default_index=0,
                styles={
                    "container": {"padding": "0!important", "background-color": "#f8f9fa"},
                    "icon": {"color": "#1f77b4", "font-size": "20px"},
                    "menu-title": {"color": "#1f77b4", "font-size": "16px", "font-weight": "bold"},
                    "menu-icon": {"display": "none"},
                    "selected": {"background-color": "#1f77b4", "color": "white", "border-radius": "5px"}
                }
            )
            
            st.markdown("---")
            
            # Información del sistema
            st.markdown("### 📊 Estado del Sistema")
            
            if st.session_state.rag_system_initialized and st.session_state.system_stats:
                stats = st.session_state.system_stats
                
                st.metric("📄 Documentos", stats.get('documents_indexed', 0))
                st.metric("🧠 Embeddings", stats.get('embeddings_model', 'N/A'))
                st.metric("🤖 LLM", stats.get('llm_model', 'N/A'))
                
                # Botón de actualización
                if st.button("🔄 Actualizar Estado"):
                    st.rerun()
            else:
                st.info("Sistema no inicializado")
                
                if st.button("🚀 Inicializar Sistema"):
                    if self.initialize_rag_system():
                        st.rerun()
            
            st.markdown("---")
            
            # Información del usuario
            st.markdown("### 👤 Usuario")
            st.write(f"**Sesión:** {datetime.now().strftime('%H:%M:%S')}")
            st.write(f"**Consultas:** {len(st.session_state.conversation_history)}")
            
            # Configuración rápida
            with st.expander("⚡ Configuración Rápida"):
                auto_clear = st.checkbox("Auto-limpiar historial", 
                                       st.session_state.user_settings['auto_clear_history'])
                show_sources = st.checkbox("Mostrar fuentes", 
                                         st.session_state.user_settings['show_sources'])
                
                if auto_clear != st.session_state.user_settings['auto_clear_history'] or \
                   show_sources != st.session_state.user_settings['show_sources']:
                    st.session_state.user_settings.update({
                        'auto_clear_history': auto_clear,
                        'show_sources': show_sources
                    })
                    st.success("Configuración actualizada")
        
        return selected
    
    def render_chat_interface(self):
        """Renderizar interfaz de chat RAG."""
        st.markdown("### 💬 Chat RAG - Consultas Inteligentes")
        
        if not st.session_state.rag_system_initialized:
            st.warning("⚠️ El sistema RAG no está inicializado. Ve a la barra lateral para inicializarlo.")
            return
        
        # Ejemplos de consultas
        st.markdown("#### 🎯 Ejemplos de consultas:")
        example_queries = [
            "¿Cuáles son las principales políticas económicas mencionadas por Milei?",
            "¿Qué dice sobre la inflación en sus discursos?",
            "¿Cuáles son sus propuestas sobre el déficit fiscal?",
            "¿Qué menciona sobre la libertad económica?",
            "¿Cuáles son sus comentarios sobre el Estado?"
        ]
        
        cols = st.columns(len(example_queries))
        for i, query in enumerate(example_queries):
            with cols[i]:
                if st.button(f"Ejemplo {i+1}", key=f"example_{i}"):
                    st.session_state.current_query = query
        
        # Input de consulta
        st.markdown("#### 💭 Haz tu consulta:")
        
        # Campo de texto principal
        query = st.text_input(
            "Escribe tu pregunta sobre los discursos de Milei:",
            key="current_query",
            placeholder="Ejemplo: ¿Qué piensa sobre la economía argentina?",
            label_visibility="collapsed"
        )
        
        # Configuración de la consulta
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            max_sources = st.slider("Número máximo de fuentes", 1, 10, 4)
        
        with col2:
            show_sources = st.checkbox("Mostrar fuentes", 
                                     st.session_state.user_settings['show_sources'])
        
        with col3:
            max_length = st.slider("Longitud máxima", 500, 3000, 2000)
        
        # Botón de consulta
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🔍 Consultar", type="primary", use_container_width=True):
                if query.strip():
                    self.process_query(query, max_sources, show_sources, max_length)
                else:
                    st.warning("⚠️ Por favor, escribe una consulta.")
        
        st.markdown("---")
        
        # Historial de conversación
        self.render_conversation_history()
    
    def process_query(self, query: str, max_sources: int, show_sources: bool, max_length: int):
        """Procesar una consulta del usuario."""
        try:
            with st.spinner("🤖 Procesando consulta..."):
                # Hacer consulta al sistema RAG
                result = self.rag_system.query(
                    question=query,
                    include_sources=show_sources,
                    max_response_length=max_length
                )
                
                # Agregar al historial
                timestamp = datetime.now()
                
                user_message = {
                    "role": "user",
                    "content": query,
                    "timestamp": timestamp,
                    "success": True
                }
                
                assistant_message = {
                    "role": "assistant", 
                    "content": result.get("answer", "Error procesando consulta"),
                    "timestamp": timestamp,
                    "success": result.get("success", False),
                    "sources": result.get("sources", []),
                    "metrics": {
                        "processing_time": result.get("processing_time_seconds", 0),
                        "confidence": result.get("confidence", 0),
                        "documents_retrieved": result.get("documents_retrieved", 0)
                    }
                }
                
                st.session_state.conversation_history.append(user_message)
                st.session_state.conversation_history.append(assistant_message)
                
                # Limpiar campo de consulta
                st.session_state.current_query = ""
                
                # Auto-limpiar si está habilitado
                if st.session_state.user_settings['auto_clear_history'] and \
                   len(st.session_state.conversation_history) > st.session_state.user_settings['max_history_items'] * 2:
                    st.session_state.conversation_history = st.session_state.conversation_history[-st.session_state.user_settings['max_history_items']*2:]
                
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Error procesando consulta: {e}")
            logger.error(f"Error en consulta: {e}")
    
    def render_conversation_history(self):
        """Renderizar historial de conversación."""
        st.markdown("### 💬 Historial de Conversación")
        
        if not st.session_state.conversation_history:
            st.info("💭 No hay conversaciones aún. ¡Haz tu primera consulta!")
            return
        
        # Mostrar conversación en orden inverso (más reciente primero)
        for i, message in enumerate(reversed(st.session_state.conversation_history)):
            message_index = len(st.session_state.conversation_history) - 1 - i
            
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 Tú:</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
                
            else:  # assistant
                # Contenido de la respuesta
                response_content = message['content']
                
                # Agregar métricas si están disponibles
                if 'metrics' in message:
                    metrics = message['metrics']
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>🤖 Sistema RAG:</strong><br>
                        {response_content}
                        <br><br>
                        <small>⏱️ {metrics.get('processing_time', 0):.2f}s | 
                        📄 {metrics.get('documents_retrieved', 0)} docs | 
                        🎯 {metrics.get('confidence', 0):.1%}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>🤖 Sistema RAG:</strong><br>
                        {response_content}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Mostrar fuentes si están disponibles
                if show_sources and 'sources' in message and message['sources']:
                    with st.expander("📚 Ver fuentes consultadas", expanded=False):
                        for j, source in enumerate(message['sources'], 1):
                            st.markdown(f"""
                            **Fuente {j}:**
                            - **Título:** {source.get('title', 'Sin título')}
                            - **Fuente:** {source.get('source', 'Desconocida')}
                            - **Tipo:** {source.get('document_type', 'N/A')}
                            - **Vista previa:** {source.get('content_preview', 'Sin contenido')}
                            """)
            
            st.markdown("")  # Espaciado
        
        # Controles del historial
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🗑️ Limpiar Historial"):
                st.session_state.conversation_history.clear()
                st.rerun()
        
        with col2:
            if st.session_state.conversation_history and \
               st.button("📥 Descargar Conversación"):
                self.download_conversation()
        
        with col3:
            if len(st.session_state.conversation_history) > 10 and \
               st.button("🔄 Cargar Más"):
                # En una implementación completa, aquí cargaríamos más historial
                st.info("📚 Funcionalidad de cargar más en desarrollo")
    
    def download_conversation(self):
        """Descargar conversación como archivo."""
        try:
            conversation_data = {
                "timestamp": datetime.now().isoformat(),
                "conversation": st.session_state.conversation_history,
                "system_info": st.session_state.system_stats
            }
            
            # Crear archivo JSON
            json_str = json.dumps(conversation_data, ensure_ascii=False, indent=2)
            st.download_button(
                label="💾 Descargar JSON",
                data=json_str,
                file_name=f"conversacion_rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"❌ Error descargando conversación: {e}")
    
    def render_documents_interface(self):
        """Renderizar interfaz de administración de documentos."""
        st.markdown("### 📁 Administración de Documentos")
        
        if not st.session_state.rag_system_initialized:
            st.warning("⚠️ El sistema RAG no está inicializado.")
            return
        
        # Pestañas para diferentes funciones
        tab1, tab2, tab3 = st.tabs(["🕷️ Web Scraping", "📤 Subir Archivos", "📊 Estado"])
        
        with tab1:
            self.render_web_scraping_tab()
        
        with tab2:
            self.render_file_upload_tab()
        
        with tab3:
            self.render_documents_status_tab()
    
    def render_web_scraping_tab(self):
        """Renderizar pestaña de web scraping."""
        st.markdown("#### 🕷️ Web Scraping - Casa Rosada")
        
        st.info("""
        **Funcionalidad:** Extraer discursos automáticamente desde el sitio web oficial de la Casa Rosada.
        
        **URL objetivo:** https://www.casarosada.gob.ar/informacion/discursos
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_discursos = st.number_input("Máximo de discursos a extraer", 
                                          min_value=1, max_value=50, value=10)
        
        with col2:
            include_content = st.checkbox("Incluir contenido completo", value=True)
        
        if st.button("🚀 Ejecutar Web Scraping", type="primary"):
            with st.spinner("🕷️ Extrayendo discursos..."):
                try:
                    result = self.rag_system.ingest_documents(
                        source="web",
                        max_documents=max_discursos,
                        force_refresh=True
                    )
                    
                    if result.get("success"):
                        st.success(f"✅ Extracción completada:")
                        st.write(f"- Documentos procesados: {result.get('documents_processed', 0)}")
                        st.write(f"- Documentos fallidos: {result.get('documents_failed', 0)}")
                        st.write(f"- Tiempo total: {result.get('processing_time_seconds', 0):.2f}s")
                        
                        # Botón para actualizar el sistema
                        if st.button("🔄 Actualizar Sistema RAG"):
                            st.rerun()
                    else:
                        st.error(f"❌ Error: {result.get('error', 'Error desconocido')}")
                        
                except Exception as e:
                    st.error(f"❌ Error inesperado: {e}")
                    logger.error(f"Error en web scraping: {e}")
    
    def render_file_upload_tab(self):
        """Renderizar pestaña de subida de archivos."""
        st.markdown("#### 📤 Subir Archivos")
        
        # Configuración de subida
        col1, col2 = st.columns(2)
        
        with col1:
            accepted_types = st.multiselect(
                "Tipos de archivo aceptados",
                [".pdf", ".txt", ".docx", ".doc"],
                default=[".pdf", ".txt", ".docx"]
            )
        
        with col2:
            max_file_size = st.slider("Tamaño máximo por archivo (MB)", 1, 50, 10)
        
        # Subida de archivos
        uploaded_files = st.file_uploader(
            "Selecciona archivos para subir:",
            type=accepted_types,
            accept_multiple_files=True,
            help="Formatos soportados: PDF, TXT, DOCX, DOC"
        )
        
        if uploaded_files:
            st.write(f"📄 {len(uploaded_files)} archivo(s) seleccionado(s)")
            
            # Mostrar información de archivos
            file_info = []
            for file in uploaded_files:
                size_mb = file.size / (1024 * 1024)
                file_info.append({
                    "Nombre": file.name,
                    "Tipo": file.type,
                    "Tamaño": f"{size_mb:.2f} MB"
                })
            
            df_files = pd.DataFrame(file_info)
            st.dataframe(df_files, use_container_width=True)
            
            # Botón de procesamiento
            if st.button("📥 Procesar Archivos", type="primary"):
                self.process_uploaded_files(uploaded_files)
    
    def process_uploaded_files(self, uploaded_files):
        """Procesar archivos subidos."""
        try:
            with st.spinner("📥 Procesando archivos..."):
                # Guardar archivos temporalmente
                temp_dir = "data/temp"
                os.makedirs(temp_dir, exist_ok=True)
                
                saved_files = []
                for file in uploaded_files:
                    file_path = os.path.join(temp_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    saved_files.append(file_path)
                
                # Procesar con el sistema RAG
                result = self.rag_system.ingest_documents(
                    source="local",
                    max_documents=len(saved_files),
                    force_refresh=True
                )
                
                if result.get("success"):
                    st.success(f"✅ Archivos procesados:")
                    st.write(f"- Archivos procesados: {result.get('documents_processed', 0)}")
                    
                    # Limpiar archivos temporales
                    for file_path in saved_files:
                        try:
                            os.remove(file_path)
                        except:
                            pass
                    
                    if st.button("🔄 Actualizar Sistema"):
                        st.rerun()
                else:
                    st.error(f"❌ Error procesando archivos: {result.get('error')}")
                    
        except Exception as e:
            st.error(f"❌ Error inesperado: {e}")
    
    def render_documents_status_tab(self):
        """Renderizar pestaña de estado de documentos."""
        st.markdown("#### 📊 Estado de Documentos")
        
        try:
            # Obtener estadísticas del sistema
            stats = self.rag_system.get_system_stats()
            
            if stats.get('error'):
                st.error(f"Error obteniendo estadísticas: {stats['error']}")
                return
            
            # Métricas principales
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{stats.get('vector_store', {}).get('document_count', 0)}</h3>
                    <p>📄 Documentos</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                embedding_dim = stats.get('vector_store', {}).get('embedding_dimension', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{embedding_dim}</h3>
                    <p>🧠 Dimensión Embeddings</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                storage_mb = stats.get('vector_store', {}).get('storage_size_mb', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{storage_mb:.1f} MB</h3>
                    <p>💾 Almacenamiento</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                collection_name = stats.get('vector_store', {}).get('collection_name', 'N/A')
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{collection_name}</h3>
                    <p>🗄️ Colección</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Información detallada
            st.markdown("#### 📋 Información Detallada")
            
            # Configuración del sistema
            st.markdown("**Configuración del Sistema:**")
            config_info = {
                "Modelo Embeddings": stats.get('configuration', {}).get('embedding_model', 'N/A'),
                "Modelo LLM": stats.get('configuration', {}).get('llm_model', 'N/A'),
                "Chunk Size": stats.get('configuration', {}).get('chunk_size', 'N/A'),
                "Top K": stats.get('configuration', {}).get('top_k', 'N/A')
            }
            
            for key, value in config_info.items():
                st.write(f"- **{key}:** {value}")
            
            # Tipos de documentos
            doc_types = stats.get('vector_store', {}).get('document_types', {})
            if doc_types:
                st.markdown("**Tipos de Documentos:**")
                for doc_type, count in doc_types.items():
                    st.write(f"- **{doc_type}:** {count}")
            
            # Acciones
            st.markdown("#### 🛠️ Acciones")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Actualizar"):
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Limpiar Base de Datos"):
                    if st.session_state.get('confirm_clear', False):
                        result = self.rag_system.clear_database()
                        if result.get('success'):
                            st.success("✅ Base de datos limpiada")
                            st.rerun()
                        else:
                            st.error(f"❌ Error: {result.get('error')}")
                    else:
                        st.session_state.confirm_clear = True
                        st.warning("⚠️ ¿Seguro? Click nuevamente para confirmar")
            
            with col3:
                if st.button("💾 Backup"):
                    self.create_backup()
        
        except Exception as e:
            st.error(f"❌ Error obteniendo estado: {e}")
    
    def create_backup(self):
        """Crear backup de la base de datos."""
        try:
            backup_path = f"backup_rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            result = self.rag_system.vector_store.backup_database(backup_path)
            
            if result.get('success'):
                st.success(f"✅ Backup creado: {backup_path}")
            else:
                st.error(f"❌ Error creando backup: {result.get('error')}")
                
        except Exception as e:
            st.error(f"❌ Error inesperado: {e}")
    
    def render_analytics_interface(self):
        """Renderizar interfaz de analytics."""
        st.markdown("### 📊 Analytics y Métricas")
        
        # Crear datos de ejemplo para gráficos
        # En una implementación real, esto vendría del sistema
        dates = pd.date_range(start='2025-11-01', end='2025-11-16', freq='D')
        queries_per_day = [5, 8, 12, 15, 10, 18, 22, 14, 16, 20, 25, 19, 23, 21, 28, 15]
        response_times = [2.1, 1.8, 2.3, 1.9, 2.0, 2.5, 2.2, 1.7, 2.1, 2.4, 2.0, 1.9, 2.3, 2.1, 2.6, 1.8]
        confidence_scores = [0.85, 0.88, 0.82, 0.90, 0.87, 0.84, 0.89, 0.91, 0.86, 0.83, 0.88, 0.90, 0.85, 0.87, 0.82, 0.89]
        
        df_analytics = pd.DataFrame({
            'fecha': dates,
            'consultas': queries_per_day,
            'tiempo_respuesta': response_times,
            'confianza': confidence_scores
        })
        
        # Gráfico de consultas por día
        st.markdown("#### 📈 Consultas por Día")
        fig_queries = px.line(
            df_analytics, 
            x='fecha', 
            y='consultas',
            title='Número de Consultas Diarias',
            labels={'consultas': 'Consultas', 'fecha': 'Fecha'}
        )
        fig_queries.update_layout(height=400)
        st.plotly_chart(fig_queries, use_container_width=True)
        
        # Métricas en tiempo real
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_queries = sum(queries_per_day)
            avg_queries = total_queries / len(queries_per_day)
            st.metric("📊 Total Consultas", total_queries, delta=f"+{avg_queries:.1f}/día")
        
        with col2:
            avg_response_time = sum(response_times) / len(response_times)
            st.metric("⏱️ Tiempo Promedio", f"{avg_response_time:.2f}s", delta="-0.3s")
        
        with col3:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            st.metric("🎯 Confianza Promedio", f"{avg_confidence:.1%}", delta="+2.1%")
        
        # Gráfico de performance
        st.markdown("#### ⚡ Performance del Sistema")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_performance = make_subplots(
                rows=1, cols=1,
                subplot_titles=['Tiempo de Respuesta'],
                specs=[[{"secondary_y": True}]]
            )
            
            fig_performance.add_trace(
                go.Scatter(x=df_analytics['fecha'], y=df_analytics['tiempo_respuesta'],
                          name='Tiempo Respuesta (s)', line=dict(color='blue')),
                secondary_y=False
            )
            
            fig_performance.update_yaxes(title_text="Segundos", secondary_y=False)
            fig_performance.update_layout(height=300, title_text="Performance Temporal")
            st.plotly_chart(fig_performance, use_container_width=True)
        
        with col2:
            fig_confidence = px.bar(
                df_analytics.tail(7), 
                x='fecha', 
                y='confianza',
                title='Confianza de Respuestas (últimos 7 días)',
                labels={'confianza': 'Confianza', 'fecha': 'Fecha'}
            )
            fig_confidence.update_yaxes(range=[0, 1])
            fig_confidence.update_layout(height=300)
            st.plotly_chart(fig_confidence, use_container_width=True)
        
        # Top consultas
        st.markdown("#### 🔥 Top Consultas")
        
        top_queries = [
            "Políticas económicas de Milei",
            "Propuestas sobre inflación",
            "Libertad económica",
            "Déficit fiscal",
            "Rol del Estado"
        ]
        
        query_data = pd.DataFrame({
            'consulta': top_queries,
            'frecuencia': [45, 38, 32, 28, 25],
            'satisfaccion': [0.92, 0.89, 0.85, 0.88, 0.90]
        })
        
        fig_top = px.bar(
            query_data,
            x='frecuencia',
            y='consulta',
            orientation='h',
            title='Consultas Más Frecuentes',
            labels={'frecuencia': 'Frecuencia', 'consulta': 'Consulta'}
        )
        fig_top.update_layout(height=400)
        st.plotly_chart(fig_top, use_container_width=True)
    
    def render_configuration_interface(self):
        """Renderizar interfaz de configuración."""
        st.markdown("### ⚙️ Configuración")
        
        # Configuración del sistema
        st.markdown("#### 🔧 Configuración del Sistema")
        
        with st.form("system_config"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Preferencias de Chat**")
                auto_clear = st.checkbox("Auto-limpiar historial", 
                                       st.session_state.user_settings['auto_clear_history'])
                show_sources = st.checkbox("Mostrar fuentes por defecto", 
                                         st.session_state.user_settings['show_sources'])
                enable_cache = st.checkbox("Habilitar cache", 
                                         st.session_state.user_settings['enable_caching'])
            
            with col2:
                st.markdown("**Configuración de Respuestas**")
                response_style = st.selectbox(
                    "Estilo de respuesta",
                    ["detailed", "summary", "simple"],
                    index=["detailed", "summary", "simple"].index(
                        st.session_state.user_settings['response_style']
                    )
                )
                max_history = st.slider("Máximo historial", 10, 100, 
                                      st.session_state.user_settings['max_history_items'])
            
            submitted = st.form_submit_button("💾 Guardar Configuración")
            
            if submitted:
                st.session_state.user_settings.update({
                    'auto_clear_history': auto_clear,
                    'show_sources': show_sources,
                    'enable_caching': enable_cache,
                    'response_style': response_style,
                    'max_history_items': max_history
                })
                st.success("✅ Configuración guardada")
        
        st.markdown("---")
        
        # Configuración de API
        st.markdown("#### 🔑 Configuración de API")
        
        with st.expander("Google Gemini API"):
            st.info("La configuración de API se maneja a través de variables de entorno.")
            
            # Mostrar estado de la API
            if st.session_state.rag_system_initialized:
                try:
                    llm_stats = self.rag_system.llm_handler.get_stats()
                    st.text(f"Estado: {'✅ Configurado' if llm_stats.get('initialized') else '❌ No disponible'}")
                    
                    if llm_stats.get('initialized'):
                        st.write(f"Modelo: {llm_stats.get('model_name')}")
                        st.write(f"Temperature: {llm_stats.get('configuration', {}).get('temperature')}")
                        st.write(f"Tasa de éxito: {llm_stats.get('statistics', {}).get('success_rate', 0):.1%}")
                    
                except Exception as e:
                    st.error(f"Error obteniendo estado de API: {e}")
        
        # Configuración de rendimiento
        st.markdown("#### ⚡ Configuración de Rendimiento")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Rate Limiting**")
            max_requests = st.number_input("Máximo requests por minuto", 
                                         min_value=1, max_value=100, value=10)
            window_seconds = st.number_input("Ventana de tiempo (segundos)", 
                                           min_value=30, max_value=3600, value=60)
            
            if st.button("🛡️ Actualizar Rate Limiting"):
                st.info(f"Rate limiting configurado: {max_requests} requests cada {window_seconds} segundos")
        
        with col2:
            st.markdown("**Cache**")
            
            # Mostrar estadísticas de cache
            if st.session_state.rag_system_initialized:
                try:
                    cache_stats = self.rag_system.embeddings_handler.get_cache_stats()
                    st.write(f"Cache en memoria: {cache_stats.get('memory_cache_size', 0)} items")
                    st.write(f"Cache en disco: {cache_stats.get('disk_cache_files', 0)} archivos")
                    st.write(f"Tamaño: {cache_stats.get('disk_cache_size_mb', 0):.2f} MB")
                    
                    if st.button("🧹 Limpiar Cache"):
                        self.rag_system.embeddings_handler.clear_cache()
                        st.success("Cache limpiado")
                        
                except Exception as e:
                    st.error(f"Error accediendo al cache: {e}")
        
        # Información del sistema
        st.markdown("#### ℹ️ Información del Sistema")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Estado de Componentes:**")
            components_status = {
                "RAG System": st.session_state.rag_system_initialized,
                "Vector Store": os.path.exists("data/vector_db"),
                "Embeddings": hasattr(self, 'embeddings_handler'),
                "Cache": st.session_state.user_settings['enable_caching']
            }
            
            for component, status in components_status.items():
                status_icon = "✅" if status else "❌"
                st.markdown(f"- {component}: {status_icon}")
        
        with col2:
            st.markdown("**Recursos del Sistema:**")
            system_info = self.utils.get_system_info()
            
            if not system_info.get("error"):
                st.write(f"- **CPU:** {system_info.get('cpu_count', 'N/A')} cores")
                st.write(f"- **Memoria:** {self.utils.format_file_size(system_info.get('memory_available', 0))} disponible")
                st.write(f"- **Disco:** {system_info.get('disk_usage', 'N/A')}% usado")
                st.write(f"- **Python:** {system_info.get('python_version', 'N/A')}")


def main():
    """Función principal de la aplicación."""
    # Crear instancia de la aplicación
    app = RAGApp()
    
    # Renderizar encabezado
    app.render_header()
    
    # Renderizar barra lateral y obtener selección
    selected_option = app.render_sidebar()
    
    # Renderizar contenido principal basado en selección
    if selected_option == "💬 Chat RAG":
        app.render_chat_interface()
    elif selected_option == "📁 Documentos":
        app.render_documents_interface()
    elif selected_option == "📊 Analytics":
        app.render_analytics_interface()
    elif selected_option == "⚙️ Configuración":
        app.render_configuration_interface()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🇦🇷 Sistema RAG - Discursos de Javier Milei | 
        Desarrollado para el TP2 de Procesamiento del Habla e Introducción a LLMs | 
        IFTS 24 - 2025</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()