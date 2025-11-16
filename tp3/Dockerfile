# Configuración de Docker para Sistema RAG - Discursos de Javier Milei
# ================================================================

FROM python:3.11-slim

# Configurar variables de entorno
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements primero (para cache de Docker)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY . .

# Crear directorios necesarios
RUN mkdir -p data/cache/embeddings data/cache/responses data/vector_db data/temp logs

# Configurar permisos
RUN chmod +x setup_streamlit.py

# Exponer puerto
EXPOSE 8501

# Comando por defecto
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Labels para metadatos
LABEL maintainer="Sistema RAG Milei"
LABEL description="Sistema RAG para análisis de discursos de Javier Milei"
LABEL version="1.0.0"
LABEL org.opencontainers.image.title="Sistema RAG - Discursos de Javier Milei"
LABEL org.opencontainers.image.description="Sistema de Recuperación y Generación Aumentada para análisis de discursos presidenciales"
LABEL org.opencontainers.image.source="https://github.com/tu-usuario/sistema-rag-milei"
LABEL org.opencontainers.image.licenses="MIT"