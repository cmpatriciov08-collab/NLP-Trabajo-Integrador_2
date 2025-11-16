"""
Web Scraper para Discursos de la Casa Rosada
===========================================

Este módulo implementa un web scraper especializado para extraer los discursos
del presidente Javier Milei desde el sitio web oficial de la Casa Rosada.

Funcionalidades:
- Scraping inteligente de la sección de discursos
- Extracción de títulos, fechas y contenido
- Limpieza automática de texto extraído
- Manejo robusto de errores y timeouts
- Respeto por las políticas del sitio web

Autor: Sistema RAG Milei
Fecha: 2025-11-16
"""

import os
import re
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

# Web scraping
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Configuración de logging
logger = logging.getLogger(__name__)


class WebScraper:
    """
    Web scraper especializado para discursos de la Casa Rosada.
    
    Extrae discursos del presidente Javier Milei desde:
    https://www.casarosada.gob.ar/informacion/discursos
    """
    
    def __init__(self, 
                 base_url: str = "https://www.casarosada.gob.ar",
                 discursos_url: str = "https://www.casarosada.gob.ar/informacion/discursos",
                 timeout: int = 10,
                 delay_between_requests: float = 1.0,
                 max_retries: int = 3):
        """
        Inicializar web scraper.
        
        Args:
            base_url: URL base del sitio web
            discursos_url: URL específica de discursos
            timeout: Timeout para requests HTTP
            delay_between_requests: Delay entre requests (segundos)
            max_retries: Máximo número de reintentos
        """
        self.base_url = base_url
        self.discursos_url = discursos_url
        self.timeout = timeout
        self.delay_between_requests = delay_between_requests
        self.max_retries = max_retries
        
        # Headers para parecer un navegador legítimo
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'es-AR,es;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Crear sesión para mantener cookies
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        logger.info("🕷️ WebScraper inicializado para Casa Rosada")
    
    def scrape_discursos(self, 
                        max_discursos: int = 10,
                        include_content: bool = True) -> List[Dict[str, Any]]:
        """
        Extraer discursos desde la Casa Rosada.
        
        Args:
            max_discursos: Máximo número de discursos a extraer
            include_content: Si incluir contenido completo de cada discurso
            
        Returns:
            Lista de diccionarios con información de discursos
        """
        try:
            logger.info(f"🎯 Iniciando scraping de {max_discursos} discursos")
            
            # 1. Obtener lista de discursos
            discursos_list = self._get_discursos_list(max_discursos)
            
            if not discursos_list:
                logger.warning("No se encontraron discursos en la página principal")
                return []
            
            # 2. Extraer contenido de cada discurso
            discursos_completos = []
            
            for i, discurso_info in enumerate(discursos_list, 1):
                logger.info(f"📄 Procesando discurso {i}/{len(discursos_list)}: {discurso_info.get('titulo', 'Sin título')[:50]}...")
                
                try:
                    if include_content:
                        contenido_completo = self._get_discurso_content(discurso_info['url'])
                        discurso_info.update(contenido_completo)
                    
                    discursos_completos.append(discurso_info)
                    
                    # Delay entre requests
                    if i < len(discursos_list):
                        time.sleep(self.delay_between_requests)
                        
                except Exception as e:
                    logger.warning(f"Error procesando discurso {discurso_info.get('titulo', 'Unknown')}: {e}")
                    # Agregar discurso con contenido básico aunque falle
                    discursos_completos.append(discurso_info)
            
            logger.info(f"✅ Scraping completado: {len(discursos_completos)} discursos extraídos")
            return discursos_completos
            
        except Exception as e:
            logger.error(f"❌ Error general en scraping: {e}")
            return []
    
    def _get_discursos_list(self, max_discursos: int) -> List[Dict[str, str]]:
        """
        Obtener lista de discursos desde la página principal.
        
        Args:
            max_discursos: Máximo número de discursos a obtener
            
        Returns:
            Lista de diccionarios con info básica de discursos
        """
        try:
            logger.info(f"🔍 Obteniendo lista de discursos desde: {self.discursos_url}")
            
            # Realizar request con reintentos
            response = self._make_request(self.discursos_url)
            if not response:
                return []
            
            # Parsear HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Buscar enlaces de discursos con múltiples estrategias
            discursos_links = []
            
            # Estrategia 1: Buscar por selectores específicos
            link_selectors = [
                'a[href*="discursos"]',
                'a[href*="discurso"]',
                '.discurso-link',
                '.titulo-discurso a',
                'h3 a',
                'h2 a',
                '.title a',
                'article a'
            ]
            
            for selector in link_selectors:
                links = soup.select(selector)
                if links:
                    logger.info(f"Found {len(links)} links with selector: {selector}")
                    break
            
            # Estrategia 2: Si no se encuentran con selectores, buscar todos los enlaces
            if not links:
                links = soup.find_all('a', href=True)
                logger.info(f"No specific selectors worked, using all {len(links)} links")
            
            # Procesar enlaces encontrados
            for link in links:
                try:
                    href = link.get('href')
                    title = link.get_text(strip=True)
                    
                    if not href or not title:
                        continue
                    
                    # Filtrar enlaces relevantes
                    if self._is_relevant_discurso_link(href, title):
                        # Construir URL completa
                        if href.startswith('/'):
                            full_url = urljoin(self.base_url, href)
                        elif href.startswith('http'):
                            full_url = href
                        else:
                            full_url = urljoin(self.base_url, href)
                        
                        # Crear entrada de discurso
                        discurso_info = {
                            'titulo': title,
                            'url': full_url,
                            'fecha_extraccion': datetime.now().isoformat(),
                            'fuente': 'Casa Rosada - Web Scraping'
                        }
                        
                        # Agregar fecha si se puede extraer del título
                        fecha = self._extract_date_from_title(title)
                        if fecha:
                            discurso_info['fecha'] = fecha
                        
                        discursos_links.append(discurso_info)
                        
                        # Limitar resultados
                        if len(discursos_links) >= max_discursos:
                            break
                            
                except Exception as e:
                    logger.warning(f"Error procesando enlace: {e}")
                    continue
            
            # Eliminar duplicados por URL
            unique_discursos = self._remove_duplicate_urls(discursos_links)
            
            logger.info(f"📋 Lista de discursos obtenida: {len(unique_discursos)} únicos")
            return unique_discursos[:max_discursos]
            
        except Exception as e:
            logger.error(f"Error obteniendo lista de discursos: {e}")
            return []
    
    def _is_relevant_discurso_link(self, href: str, title: str) -> bool:
        """
        Determinar si un enlace es relevante para discursos.
        
        Args:
            href: URL del enlace
            title: Título del enlace
            
        Returns:
            True si el enlace es relevante
        """
        # Criterios de relevancia
        relevance_criteria = [
            # Contenido del href
            'discursos' in href.lower(),
            'discurso' in href.lower(),
            'informacion' in href.lower(),
            
            # Contenido del título
            any(word in title.lower() for word in [
                'discurso', 'milei', 'presidente', 'palabras', 'alocución',
                'mensaje', 'javier', 'conferencia', 'declaración', 'ministro',
                'argentina', 'nación', 'gobierno', 'política', 'economía'
            ]),
            
            # Patrones de fecha en el título
            re.search(r'\d{1,2}\s+de\s+\w+', title) is not None,
            
            # Longitud mínima del título
            len(title) > 10
        ]
        
        return any(relevance_criteria)
    
    def _remove_duplicate_urls(self, discursos: List[Dict]) -> List[Dict]:
        """Eliminar discursos duplicados por URL."""
        seen_urls = set()
        unique_discursos = []
        
        for discurso in discursos:
            # Normalizar URL para comparación
            url_normalizada = discurso['url'].split('?')[0]  # Remover parámetros
            
            if url_normalizada not in seen_urls:
                unique_discursos.append(discurso)
                seen_urls.add(url_normalizada)
        
        return unique_discursos
    
    def _extract_date_from_title(self, title: str) -> Optional[str]:
        """Extraer fecha del título del discurso."""
        # Patrones de fecha en títulos
        date_patterns = [
            r'(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})',  # "10 de enero de 2024"
            r'(\d{1,2})/(\d{1,2})/(\d{4})',            # "10/01/2024"
            r'(\d{4})-(\d{2})-(\d{2})',                # "2024-01-10"
        ]
        
        meses = {
            'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04',
            'mayo': '05', 'junio': '06', 'julio': '07', 'agosto': '08',
            'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12'
        }
        
        for pattern in date_patterns:
            match = re.search(pattern, title.lower())
            if match:
                try:
                    if 'de' in pattern:  # Formato "10 de enero de 2024"
                        day, month_name, year = match.groups()
                        month = meses.get(month_name, '01')
                    else:  # Formato numérico
                        parts = match.groups()
                        if len(parts[0]) == 4:  # YYYY-MM-DD
                            year, month, day = parts
                        else:  # DD/MM/YYYY
                            day, month, year = parts
                    
                    date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                    datetime.strptime(date_str, "%Y-%m-%d")
                    return date_str
                    
                except ValueError:
                    continue
        
        return None
    
    def _get_discurso_content(self, discurso_url: str) -> Dict[str, Any]:
        """
        Extraer contenido completo de un discurso específico.
        
        Args:
            discurso_url: URL del discurso
            
        Returns:
            Diccionario con contenido del discurso
        """
        try:
            # Realizar request
            response = self._make_request(discurso_url)
            if not response:
                return {'contenido': '', 'fecha_publicacion': 'No disponible'}
            
            # Parsear HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extraer contenido con múltiples estrategias
            content_selectors = [
                '.contenido',
                '.content',
                '.texto',
                'article',
                '.post-content',
                '.entry-content',
                '.field-body',
                'main',
                '.node-content'
            ]
            
            content_element = None
            for selector in content_selectors:
                content_element = soup.select_one(selector)
                if content_element:
                    break
            
            # Si no se encuentra con selectores, buscar el elemento con más texto
            if not content_element:
                text_elements = soup.find_all(['div', 'section', 'article', 'main'])
                if text_elements:
                    content_element = max(text_elements, 
                                        key=lambda x: len(x.get_text(strip=True)))
            
            # Extraer texto
            if content_element:
                texto_completo = content_element.get_text(separator='\n', strip=True)
            else:
                # Fallback: extraer todo el texto del body
                body = soup.find('body')
                texto_completo = body.get_text(separator='\n', strip=True) if body else ""
            
            # Extraer fecha de publicación
            fecha_publicacion = self._extract_publication_date(soup)
            
            # Limpiar texto
            texto_limpio = self._clean_extracted_text(texto_completo)
            
            return {
                'contenido': texto_limpio,
                'fecha_publicacion': fecha_publicacion,
                'url_original': discurso_url,
                'contenido_crudo': texto_completo
            }
            
        except Exception as e:
            logger.warning(f"Error extrayendo contenido de {discurso_url}: {e}")
            return {
                'contenido': '',
                'fecha_publicacion': 'Error en extracción',
                'url_original': discurso_url,
                'error': str(e)
            }
    
    def _extract_publication_date(self, soup: BeautifulSoup) -> str:
        """Extraer fecha de publicación del discurso."""
        date_selectors = [
            'time',
            '.fecha',
            '.date',
            '.fecha-publicacion',
            '.entry-date',
            '.post-date',
            '.publish-date',
            '.created',
            '[datetime]'
        ]
        
        for selector in date_selectors:
            date_element = soup.select_one(selector)
            if date_element:
                date_text = date_element.get('datetime') or date_element.get_text(strip=True)
                if date_text:
                    return date_text
        
        # Buscar en el contenido de texto
        text_content = soup.get_text()
        date_match = re.search(r'\d{1,2}\s+de\s+\w+\s+de\s+\d{4}', text_content)
        if date_match:
            return date_match.group()
        
        return 'Fecha no disponible'
    
    def _clean_extracted_text(self, text: str) -> str:
        """Limpiar texto extraído del HTML."""
        if not text:
            return ""
        
        # Normalizar saltos de línea
        text = re.sub(r'\r\n|\r', '\n', text)
        
        # Eliminar espacios múltiples
        text = re.sub(r' +', ' ', text)
        
        # Eliminar líneas muy cortas que puedan ser ruido
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Filtrar líneas demasiado cortas (excepto números y fechas)
        filtered_lines = []
        for line in lines:
            if (len(line) > 5 or 
                re.match(r'\d{1,2}\s+de\s+\w+', line) or
                re.match(r'\d{1,2}/\d{1,2}/\d{4}', line) or
                any(char in line for char in ['.', '!', '?', ':', ';'])):
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def _make_request(self, url: str) -> Optional[requests.Response]:
        """
        Realizar request HTTP con reintentos y manejo de errores.
        
        Args:
            url: URL a solicitar
            
        Returns:
            Response object o None si falla
        """
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Request attempt {attempt + 1}: {url}")
                
                response = self.session.get(
                    url,
                    timeout=self.timeout,
                    allow_redirects=True
                )
                
                response.raise_for_status()
                return response
                
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                
        logger.error(f"All retry attempts failed for: {url}")
        return None
    
    def get_scraping_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del scraper."""
        return {
            "base_url": self.base_url,
            "discursos_url": self.discursos_url,
            "timeout": self.timeout,
            "delay_between_requests": self.delay_between_requests,
            "max_retries": self.max_retries,
            "headers": self.headers,
            "session_active": self.session is not None
        }