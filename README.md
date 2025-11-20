---
title: MVP Consultoría RAG para Discursos de Javier Milei
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.28.1"
app_file: app.py
pinned: false
---

# MVP Consultoría RAG para Discursos de Javier Milei

LINK: https://huggingface.co/spaces/manuelcpv92/mvp_ag_optimizado

Este proyecto es un Mínimo Producto Viable (MVP) de un sistema de Generación Aumentada por Recuperación (RAG) diseñado para consultar discursos oficiales y transcripciones de audios del Presidente Javier Milei. Utiliza técnicas de procesamiento de lenguaje natural para permitir consultas precisas sobre el contenido de sus discursos públicos y transcripciones.

## Descripción

El sistema extrae automáticamente discursos oficiales desde el sitio web de Casa Rosada y permite la adición manual de transcripciones de audios, procesa todo el contenido y lo indexa en una base de datos vectorial. Actualmente incluye discursos oficiales y 3 transcripciones de audios agregadas. Luego, mediante un modelo de lenguaje (Google Gemini), responde preguntas específicas basadas únicamente en la información contenida en estos documentos.

## Funcionalidades

- **Extracción automática de discursos**: Scraping inteligente del sitio web oficial de Casa Rosada para obtener discursos recientes.
- **Adición manual de transcripciones**: Incorporación de transcripciones de audios para enriquecer el corpus (actualmente incluye 3 transcripciones agregadas).
- **Procesamiento de texto**: Limpieza y estructuración del contenido de los discursos y transcripciones.
- **Indexación vectorial**: Creación de embeddings multilingües para búsqueda semántica eficiente.
- **Consulta RAG**: Sistema de preguntas y respuestas que combina recuperación de información con generación de respuestas.
- **Interfaz web**: Aplicación Streamlit intuitiva para realizar consultas.
- **Fuentes verificadas**: Cada respuesta incluye referencias a los discursos o transcripciones específicos utilizados.

## Arquitectura del sistema RAG

El sistema RAG implementado sigue un flujo estándar de Generación Aumentada por Recuperación:

1. **Ingesta de datos**: Extracción automática de discursos desde el sitio web oficial de Casa Rosada mediante scraping y adición manual de transcripciones de audios.
2. **Procesamiento de texto**: Limpieza, tokenización y división en chunks (fragmentos) para optimizar la indexación.
3. **Indexación vectorial**: Conversión de los chunks a embeddings numéricos utilizando un modelo de embeddings multilingüe.
4. **Almacenamiento**: Persistencia de los embeddings en una base de datos vectorial para búsquedas eficientes.
5. **Consulta**: Procesamiento de la pregunta del usuario, búsqueda de chunks relevantes y generación de respuesta basada en el contexto recuperado.

### Diagrama de flujo

```
Scraping de discursos (generate_corpus.py) --> Procesamiento de texto --> Creación de embeddings (mvp_rag.py) --> Almacenamiento vectorial
Adición de transcripciones (add_transcripts.py) --> Procesamiento de texto
Consulta del usuario (app.py) --> Búsqueda de documentos relevantes --> Generación de respuesta --> Respuesta con fuentes
```

## Decisiones de diseño

- **Separación de ingesta**: El proceso de scraping y generación del corpus se realizó en un script independiente (`generate_corpus.py`) para mantener la modularidad, facilitar pruebas y permitir actualizaciones del dataset sin afectar el núcleo del sistema RAG.
- **Modelo de embeddings**: Se seleccionó `intfloat/multilingual-e5-large` por su excelente rendimiento en tareas multilingües, especialmente en español, y su capacidad para capturar semántica contextual en textos largos.
- **Modelo de lenguaje**: Google Gemini 2.5 Flash fue elegido por su velocidad de respuesta, bajo costo y capacidad nativa para procesar consultas en español sin necesidad de fine-tuning adicional.
- **Base de datos vectorial**: ChromaDB se utilizó por su simplicidad de integración con Python, persistencia local y soporte eficiente para búsquedas de similitud coseno.
- **Estrategia de chunking adaptativa**: Se implementó una clasificación automática de documentos basada en su longitud para optimizar el chunking. Documentos largos (>5000 caracteres, discursos oficiales) usan chunk_size=1000 con overlap=200, mientras que documentos cortos (<1000 caracteres, transcripciones) usan chunk_size=300 con overlap=50. Longitudes intermedias usan la configuración fallback de 500/50, preservando la coherencia contextual mediante separadores naturales.
- **Interfaz de usuario**: Streamlit se implementó para una interfaz web simple y accesible, priorizando la usabilidad sobre complejidad, adecuado para un MVP.
- **Fuentes verificadas**: Se incluyó un sistema de referencias para cada respuesta, asegurando transparencia y permitiendo verificación de la información utilizada.

## Cómo Usar la App

### Requisitos Previos

- Python 3.8+
- Clave de API de Google (Google AI Studio)
- Conexión a internet para scraping inicial

### Instalación

1. Clona o descarga el proyecto.
2. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```
3. Configura la variable de entorno:
   ```
   export GOOGLE_API_KEY="tu_clave_api_aqui"
   ```
   O en Windows:
   ```
   set GOOGLE_API_KEY=tu_clave_api_aqui
   ```

### Ejecución

1. Ejecuta la aplicación:
   ```
   streamlit run app.py
   ```
2. Abre el navegador en la URL proporcionada (generalmente http://localhost:8501).
3. Espera a que se inicialice el sistema RAG (puede tomar unos minutos en la primera ejecución).
4. Ingresa tu pregunta en el campo de texto y haz clic en "Consultar".
5. Revisa la respuesta y las fuentes consultadas.

### Ejemplos de Consultas

- "¿Qué dice Milei sobre la economía?"
- "¿Cuáles son las prioridades del gobierno según los discursos?"
- "¿Qué menciona sobre educación?"

## Despliegue en HF Spaces

https://huggingface.co/spaces/manuelcpv92/Consultor_Virtual_RAG_de_Politicas_y_Discursos_de_Milei

### Preparación

1. Crea una cuenta en [Hugging Face](https://huggingface.co).
2. Crea un nuevo Space con la opción "Streamlit".
3. Sube los archivos del proyecto (app.py, mvp_rag.py, requirements.txt).
4. Para el vectorstore, tienes dos opciones:
   - **Regenerar en HF**: El sistema scrapeará y creará el vectorstore al iniciar (recomendado para datos actualizados).
   - **Subir chroma_db**: Sube la carpeta chroma_db pre-generada (útil para consistencia).

### Configuración de Secrets

En la configuración del Space, agrega el secret:
- `GOOGLE_API_KEY`: Tu clave de API de Google.

### Notas sobre Despliegue

- El scraping inicial puede tomar tiempo; considera aumentar el timeout si es necesario.
- HF Spaces tiene límites de recursos; para uso intensivo, considera instancias pagas.
- El vectorstore se regenera en cada reinicio; para persistencia, implementa almacenamiento en HF Datasets.
- Asegúrate de que el modelo de embeddings sea compatible con los recursos de HF.

## Notas Técnicas

- Utiliza ChromaDB para almacenamiento vectorial.
- Embeddings multilingües con `intfloat/multilingual-e5-large`.
- Modelo LLM: Google Gemini 2.5 Flash.
- Solo utiliza información de discursos oficiales verificados y transcripciones de audios agregadas.
- El sistema está optimizado para consultas en español.

## Limitaciones

- Depende de la disponibilidad del sitio web de Casa Rosada.
- Respuestas limitadas a la información presente en los discursos indexados.
- No incluye discursos anteriores a la implementación del scraper ni transcripciones no agregadas manualmente.

## Estructura de Archivos

```
PROYECTO C/
├── app.py                 # Aplicación principal de Streamlit para la interfaz web
├── mvp_rag.py            # Módulo principal con lógica RAG, embeddings y configuración del sistema
├── generate_corpus.py    # Script para scraping y generación automática del corpus de discursos
├── add_transcripts.py    # Script para agregar transcripciones manualmente al corpus
├── mi_corpus.json        # Archivo JSON que contiene el corpus completo de discursos procesados
├── requirements.txt      # Lista de dependencias de Python con versiones específicas
├── chroma_db/            # Directorio de la base de datos vectorial ChromaDB
│   ├── chroma.sqlite3    # Base de datos SQLite para metadatos
│   ├── [uuid]/           # Directorio con archivos de datos vectoriales
│   │   ├── data_level0.bin
│   │   ├── header.bin
│   │   ├── length.bin
│   │   └── link_lists.bin
└── README.md             # Documentación del proyecto
```

## Dependencias y Versiones

El proyecto utiliza las siguientes bibliotecas principales:

- **LangChain (0.1.20)**: Framework para construir aplicaciones con LLMs
- **LangChain Google GenAI**: Integración con modelos de Google Gemini
- **ChromaDB**: Base de datos vectorial para almacenamiento de embeddings
- **Sentence Transformers**: Para generar embeddings multilingües con `intfloat/multilingual-e5-large`
- **Streamlit (1.28.1)**: Framework para la interfaz web
- **BeautifulSoup4**: Para parsing HTML en el scraping
- **Requests**: Para realizar peticiones HTTP
- **Pandas**: Para manipulación de datos durante el procesamiento

## Decisiones Técnicas Detalladas

### Arquitectura RAG
- **Retriever**: Usa Maximum Marginal Relevance (MMR) con k=5 documentos principales y fetch_k=15 para diversidad
- **Chunking adaptativo**: Clasificación automática por longitud - largos (>5000 chars): 1000/200, cortos (<1000 chars): 300/50, intermedios: 500/50, separados por párrafos naturales
- **Embeddings**: Modelo `intfloat/multilingual-e5-large` optimizado para español y contexto largo
- **LLM**: Gemini 2.5 Flash con temperatura 0.1 para respuestas consistentes
- **Memoria**: ConversationBufferMemory para mantener contexto en conversaciones

### Procesamiento de Datos
- **Scraping**: Automatizado desde sitio oficial de Casa Rosada
- **Limpieza**: Eliminación de HTML, normalización de texto
- **Metadata**: Conserva título, fecha y URL de cada discurso
- **Persistencia**: Corpus en JSON + vectorstore en ChromaDB local

### Interfaz y UX
- **Streamlit**: Elegido por simplicidad y deployment en HF Spaces
- **Historial**: Mantiene conversación completa en session state
- **Fuentes**: Expansores para mostrar fragmentos relevantes con metadata

## Desarrollo y Contribución

### Configuración del Entorno de Desarrollo

1. Clona el repositorio:
   ```
   git clone <url-del-repositorio>
   cd PROYECTO\ C
   ```

2. Crea un entorno virtual:
   ```
   python -m venv venv
   # En Windows:
   venv\Scripts\activate
   # En Linux/Mac:
   source venv/bin/activate
   ```

3. Instala dependencias:
   ```
   pip install -r requirements.txt
   ```

4. Configura la API key:
   ```
   export GOOGLE_API_KEY="tu_clave_api_aqui"
   # En Windows:
   set GOOGLE_API_KEY=tu_clave_api_aqui
   ```

### Ejecutar Componentes Individuales

- **Generar corpus desde web**: `python generate_corpus.py`
- **Agregar transcripciones manualmente**: `python add_transcripts.py`
- **Ejecutar aplicación web**: `streamlit run app.py`
- **Probar sistema RAG**: Importar funciones desde `mvp_rag.py`

### Guías de Contribución

1. **Fork** el proyecto
2. Crea una **rama** para tu feature: `git checkout -b feature/nueva-funcionalidad`
3. **Commit** tus cambios: `git commit -m 'Agrega nueva funcionalidad'`
4. **Push** a la rama: `git push origin feature/nueva-funcionalidad`
5. Abre un **Pull Request**

### Áreas de Mejora Sugeridas

- Implementar tests unitarios para funciones críticas
- Agregar logging detallado para debugging
- Optimizar rendimiento del vectorstore para datasets más grandes
- Implementar cache para respuestas frecuentes
- Mejorar el prompt engineering para respuestas más precisas
- Agregar soporte para múltiples idiomas
- Implementar evaluación automática de calidad de respuestas

### Reportar Issues

Para reportar bugs o solicitar nuevas funcionalidades:

1. Verifica que el issue no exista ya
2. Proporciona detalles completos: pasos para reproducir, entorno, logs de error
3. Incluye ejemplos de consultas que fallan o comportamientos inesperados
4. Etiqueta apropiadamente (bug, enhancement, question)

## Licencia

Este proyecto es de código abierto. Consulta el archivo LICENSE para más detalles.

## 👤 Grupo

**[Velasquez Christian]**  
- Email: 94721647@ifts24.edu.ar

- **[Sanchez Carlos]**  
- Email: 18253606@ifts24.edu.ar

**Trabajo Integrador - NLP**  
Fecha de realización: [20/11/25]
---





