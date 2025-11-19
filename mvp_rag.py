import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime
import time
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings


def crear_vectorstore(discursos):
    # Configuraciones de splitters
    splitter_largo = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    splitter_corto = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    splitter_fallback = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )

    documentos_langchain = []
    fragmentos = []
    for doc in discursos:
        documento = Document(
            page_content=doc["contenido_limpio"],
            metadata={
                "titulo": doc["titulo"],
                "fecha_publicacion": doc["fecha_publicacion"],
                "url": doc["url"]
            }
        )
        documentos_langchain.append(documento)

        # Clasificar documento basado en longitud del contenido
        contenido_len = len(doc["contenido_limpio"])
        if contenido_len > 5000:
            # Documentos largos/técnicos (discursos oficiales)
            splitter = splitter_largo
        elif contenido_len < 1000:
            # Documentos cortos (transcripciones)
            splitter = splitter_corto
        else:
            # Fallback para longitudes intermedias
            splitter = splitter_fallback

        frags = splitter.split_documents([documento])
        fragmentos.extend(frags)

    print("DEBUG: Cargando embeddings...")
    embeddings = SentenceTransformerEmbeddings(model_name="intfloat/multilingual-e5-large")

    print(f"DEBUG: Creando vectorstore con {len(fragmentos)} fragmentos...")
    vectorstore = Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings,
        collection_name="documentos_discursos",
        persist_directory="./chroma_db"
    )

    print("DEBUG: Vectorstore creado y persistido")
    return vectorstore

def configurar_rag(vectorstore):
    print("DEBUG: Configurando LLM...")
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("DEBUG: GOOGLE_API_KEY no encontrada")
        raise ValueError("GOOGLE_API_KEY no configurada")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=api_key
    )
    print("DEBUG: LLM configurado")

    template = """
    Sos un asistente especializado en discursos y políticas del presidente Javier Milei.
    Tu misión es proporcionar información precisa y útil basada ÚNICAMENTE en
    los discursos oficiales disponibles.

    INSTRUCCIONES IMPORTANTES:
    1. Solo usá información que aparece EXPLÍCITAMENTE en los discursos
    2. Si no encontrás la información específica, indicá claramente "No se menciona en los discursos disponibles"
    3. Sé preciso con nombres, fechas, cifras y datos específicos
    4. Usá un tono amigable pero informado y profesional

    FORMATO DE RESPUESTA:
    📋 Comenzá con un resumen ejecutivo al inicio de la respuesta
    📅 Proporcioná detalles específicos (nombres, fechas, eventos, cifras)
    🔢 Si hay múltiples puntos, numerálos claramente
    📜 Mencioná en qué discurso y fecha se encuentra cada información
    ❓ Si algo no está claro o no aparece en los documentos, decilo explícitamente
    ✅ Finalizá con una nota sobre la confiabilidad basada en las fuentes utilizadas

    DISCURSOS DISPONIBLES:
    {context}

    CONSULTA:
    {question}

    RESPUESTA ESTRUCTURADA:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15, "lambda_mult": 0.6}
    )

    sistema_rag = RetrievalQA.from_llm(
        llm=llm,
        retriever=retriever,
        prompt=prompt,
        return_source_documents=True
    )

    return sistema_rag

def consultar_rag(sistema_rag, pregunta):
    print(f"DEBUG: Consultando RAG con pregunta: {pregunta}")
    resultado = sistema_rag({"query": pregunta})
    print(f"DEBUG: Resultado de RAG: {resultado}")
    print(f"DEBUG: Tipo de resultado: {type(resultado)}")
    print(f"DEBUG: Keys en resultado: {list(resultado.keys())}")
    print(f"DEBUG: Resultado 'result': {resultado.get('result', 'No result key')[:200]}...")
    print(f"DEBUG: Resultado 'source_documents': {len(resultado.get('source_documents', []))} documentos")
    return resultado
def load_discursos_from_file():
    try:
        with open('mi_corpus.json', 'r', encoding='utf-8') as f:
            discursos = json.load(f)
        print(f"DEBUG: Discursos cargados desde archivo: {len(discursos)}")
        return discursos
    except FileNotFoundError:
        raise FileNotFoundError("El archivo 'mi_corpus.json' no se encuentra.")
    except json.JSONDecodeError:
        raise ValueError("El archivo 'mi_corpus.json' no es un JSON válido.")