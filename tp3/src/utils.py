"""
Utilidades del Sistema RAG - Discursos de Javier Milei
====================================================

Este módulo contiene funciones utilitarias y herramientas de apoyo
para el sistema RAG completo.

Funcionalidades:
- Utilidades de archivos y directorios
- Funciones de validación y sanitización
- Herramientas de logging y monitoreo
- Funciones de formateo y conversión
- Utilidades de fechas y tiempo
- Helpers para configuración y entorno

Autor: Sistema RAG Milei
Fecha: 2025-11-16
"""

import os
import re
import json
import logging
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import unicodedata
import mimetypes

# Configuración de logging
logger = logging.getLogger(__name__)


class Utils:
    """
    Clase de utilidades para el sistema RAG.
    
    Proporciona funciones helper para operaciones comunes
    en el sistema RAG de discursos de Milei.
    """
    
    @staticmethod
    def sanitize_filename(filename: str, max_length: int = 100) -> str:
        """
        Sanitizar nombre de archivo para uso seguro en filesystem.
        
        Args:
            filename: Nombre original del archivo
            max_length: Longitud máxima del nombre sanitizado
            
        Returns:
            Nombre de archivo seguro
        """
        # Normalizar caracteres Unicode
        filename = unicodedata.normalize('NFKD', filename)
        
        # Remover caracteres no ASCII
        filename = filename.encode('ascii', 'ignore').decode('ascii')
        
        # Remover caracteres especiales peligrosos
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Reemplazar espacios múltiples con guión bajo
        filename = re.sub(r'\s+', '_', filename)
        
        # Remover guiones bajos al inicio y final
        filename = filename.strip('_')
        
        # Limitar longitud
        if len(filename) > max_length:
            name, ext = os.path.splitext(filename)
            filename = name[:max_length - len(ext)] + ext
        
        # Evitar nombres reservados de Windows
        reserved_names = ['CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 
                         'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 
                         'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9']
        
        if filename.upper() in reserved_names:
            filename = f"{filename}_file"
        
        return filename or "unnamed_file"
    
    @staticmethod
    def ensure_directory(directory: str) -> bool:
        """
        Crear directorio si no existe.
        
        Args:
            directory: Ruta del directorio
            
        Returns:
            True si el directorio existe o se creó exitosamente
        """
        try:
            os.makedirs(directory, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error creando directorio {directory}: {e}")
            return False
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """
        Obtener tamaño de archivo en bytes.
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            Tamaño en bytes o 0 si hay error
        """
        try:
            return os.path.getsize(file_path)
        except Exception:
            return 0
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """
        Formatear tamaño de archivo en formato legible.
        
        Args:
            size_bytes: Tamaño en bytes
            
        Returns:
            String formateado (e.g., "1.5 MB")
        """
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"
    
    @staticmethod
    def get_file_mime_type(file_path: str) -> str:
        """
        Obtener tipo MIME de archivo.
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            Tipo MIME o 'application/octet-stream' por defecto
        """
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or 'application/octet-stream'
    
    @staticmethod
    def get_local_documents(directory: str, 
                           extensions: Optional[List[str]] = None) -> List[str]:
        """
        Obtener lista de archivos de documentos en un directorio.
        
        Args:
            directory: Directorio a explorar
            extensions: Extensiones de archivo a incluir
            
        Returns:
            Lista de rutas de archivos encontrados
        """
        if extensions is None:
            extensions = ['.txt', '.pdf', '.docx', '.doc']
        
        if not os.path.exists(directory):
            return []
        
        documents = []
        
        try:
            for file_path in Path(directory).rglob('*'):
                if file_path.is_file():
                    if file_path.suffix.lower() in extensions:
                        documents.append(str(file_path))
        except Exception as e:
            logger.error(f"Error explorando directorio {directory}: {e}")
        
        return sorted(documents)
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validar formato de email.
        
        Args:
            email: Email a validar
            
        Returns:
            True si el email es válido
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """
        Validar formato de URL.
        
        Args:
            url: URL a validar
            
        Returns:
            True si la URL es válida
        """
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return re.match(pattern, url) is not None
    
    @staticmethod
    def extract_domain(url: str) -> str:
        """
        Extraer dominio de una URL.
        
        Args:
            url: URL completa
            
        Returns:
            Dominio extraído
        """
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc
        except Exception:
            return ""
    
    @staticmethod
    def generate_unique_id(length: int = 16) -> str:
        """
        Generar ID único basado en timestamp y hash.
        
        Args:
            length: Longitud del ID
            
        Returns:
            ID único como string
        """
        timestamp = datetime.now().timestamp()
        random_data = os.urandom(16)
        combined = f"{timestamp}{random_data}".encode()
        return hashlib.sha256(combined).hexdigest()[:length]
    
    @staticmethod
    def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        """
        Formatear datetime en string legible.
        
        Args:
            dt: Datetime a formatear
            format_str: Formato de salida
            
        Returns:
            String formateado
        """
        try:
            return dt.strftime(format_str)
        except Exception:
            return str(dt)
    
    @staticmethod
    def parse_datetime(date_str: str, formats: Optional[List[str]] = None) -> Optional[datetime]:
        """
        Parsear string a datetime usando múltiples formatos.
        
        Args:
            date_str: String de fecha
            formats: Lista de formatos a intentar
            
        Returns:
            Datetime object o None si no se puede parsear
        """
        if formats is None:
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
                "%d/%m/%Y",
                "%d-%m-%Y",
                "%Y/%m/%d",
                "%d %B %Y",
                "%d de %B de %Y"
            ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        
        # Intentar parsing automático como último recurso
        try:
            from dateutil import parser
            return parser.parse(date_str)
        except Exception:
            return None
    
    @staticmethod
    def get_relative_time(dt: datetime) -> str:
        """
        Obtener tiempo relativo desde un datetime.
        
        Args:
            dt: Datetime de referencia
            
        Returns:
            String con tiempo relativo
        """
        now = datetime.now()
        diff = now - dt
        
        if diff.days > 365:
            years = diff.days // 365
            return f"hace {years} año{'s' if years > 1 else ''}"
        elif diff.days > 30:
            months = diff.days // 30
            return f"hace {months} mes{'es' if months > 1 else ''}"
        elif diff.days > 0:
            return f"hace {diff.days} día{'s' if diff.days > 1 else ''}"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"hace {hours} hora{'s' if hours > 1 else ''}"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"hace {minutes} minuto{'s' if minutes > 1 else ''}"
        else:
            return "hace un momento"
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
        """
        Truncar texto con sufijo opcional.
        
        Args:
            text: Texto a truncar
            max_length: Longitud máxima
            suffix: Sufijo para texto truncado
            
        Returns:
            Texto truncado
        """
        if len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Limpiar texto básico removiendo caracteres problemáticos.
        
        Args:
            text: Texto a limpiar
            
        Returns:
            Texto limpio
        """
        if not text:
            return ""
        
        # Normalizar Unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Remover caracteres de control
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
        
        # Normalizar saltos de línea
        text = re.sub(r'\r\n|\r', '\n', text)
        
        # Remover espacios múltiples
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def validate_api_key(api_key: str, service: str) -> bool:
        """
        Validar API key básica.
        
        Args:
            api_key: API key a validar
            service: Servicio (google, openai, etc.)
            
        Returns:
            True si la API key parece válida
        """
        if not api_key or len(api_key) < 10:
            return False
        
        patterns = {
            'google': r'^[A-Za-z0-9_-]{20,}$',
            'openai': r'^sk-[A-Za-z0-9]{48}$',
            'anthropic': r'^sk-ant-[A-Za-z0-9-]{95}$'
        }
        
        pattern = patterns.get(service.lower())
        if pattern:
            return re.match(pattern, api_key) is not None
        
        return True
    
    @staticmethod
    def load_json_config(config_path: str, default: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Cargar configuración JSON desde archivo.
        
        Args:
            config_path: Ruta del archivo de configuración
            default: Configuración por defecto si el archivo no existe
            
        Returns:
            Diccionario de configuración
        """
        if default is None:
            default = {}
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return default
        except Exception as e:
            logger.error(f"Error cargando configuración desde {config_path}: {e}")
            return default
    
    @staticmethod
    def save_json_config(config: Dict[str, Any], config_path: str) -> bool:
        """
        Guardar configuración JSON a archivo.
        
        Args:
            config: Diccionario de configuración
            config_path: Ruta del archivo de configuración
            
        Returns:
            True si se guardó exitosamente
        """
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error guardando configuración en {config_path}: {e}")
            return False
    
    @staticmethod
    def setup_logging(level: str = "INFO", 
                     log_file: Optional[str] = None,
                     format_str: Optional[str] = None) -> logging.Logger:
        """
        Configurar sistema de logging.
        
        Args:
            level: Nivel de logging
            log_file: Archivo de log (opcional)
            format_str: Formato personalizado
            
        Returns:
            Logger configurado
        """
        if format_str is None:
            format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Configurar nivel
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        
        # Configurar logging
        logging.basicConfig(
            level=numeric_level,
            format=format_str,
            handlers=[
                logging.StreamHandler(),  # Console
                logging.FileHandler(log_file, encoding='utf-8') if log_file else logging.NullHandler()
            ]
        )
        
        return logging.getLogger(__name__)
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """
        Obtener información del sistema.
        
        Returns:
            Diccionario con información del sistema
        """
        import platform
        import psutil
        
        try:
            return {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "disk_usage": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent,
                "working_directory": os.getcwd()
            }
        except Exception as e:
            logger.error(f"Error obteniendo info del sistema: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def benchmark_function(func, *args, **kwargs) -> Dict[str, Any]:
        """
        Benchmark de una función.
        
        Args:
            func: Función a benchmarkear
            *args: Argumentos posicionales
            **kwargs: Argumentos keyword
            
        Returns:
            Diccionario con resultados del benchmark
        """
        import time
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        return {
            "result": result,
            "execution_time_seconds": end_time - start_time,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def retry_function(func, 
                      max_retries: int = 3, 
                      delay: float = 1.0, 
                      backoff: float = 2.0,
                      *args, **kwargs) -> Any:
        """
        Ejecutar función con reintentos automáticos.
        
        Args:
            func: Función a ejecutar
            max_retries: Máximo número de reintentos
            delay: Delay inicial entre reintentos
            backoff: Factor de multiplicación del delay
            *args: Argumentos posicionales
            **kwargs: Argumentos keyword
            
        Returns:
            Resultado de la función o raise exception
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    sleep_time = delay * (backoff ** attempt)
                    logger.warning(f"Intento {attempt + 1} falló: {e}. Reintentando en {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Función falló después de {max_retries + 1} intentos")
        
        raise last_exception