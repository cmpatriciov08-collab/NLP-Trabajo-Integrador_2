import streamlit as st
import nest_asyncio
from datetime import datetime
nest_asyncio.apply()
from mvp_rag import load_discursos_from_file, crear_vectorstore, configurar_rag, consultar_rag

st.set_page_config(page_title="RAG Consultor de Discursos Milei", page_icon="🇦🇷")

st.title("🇦🇷 Consultor RAG de Discursos del Presidente Javier Milei")
st.markdown("Sistema de Generación Aumentada por Recuperación para consultar discursos oficiales.")

# Inicializar el sistema RAG
@st.cache_resource
def inicializar_sistema():
    st.write("Inicializando sistema RAG... Esto puede tomar unos minutos.")

    import os
    if not os.getenv('GOOGLE_API_KEY'):
        st.error("GOOGLE_API_KEY no configurada. Configura la variable de entorno GOOGLE_API_KEY con tu clave de API de Google.")
        return None, None

    # Cargar discursos desde archivo
    try:
        discursos = load_discursos_from_file()
        print(f"DEBUG: Discursos cargados: {len(discursos)}")
    except Exception as e:
        print(f"DEBUG: Error cargando discursos: {str(e)}")
        st.error(f"Error al cargar discursos: {str(e)}")
        return None, None

    if not discursos:
        print("DEBUG: No se encontraron discursos")
        st.error("No se encontraron discursos en el archivo.")
        return None, None

    st.write(f"Se cargaron {len(discursos)} discursos desde el archivo exitosamente.")

    # Crear vectorstore
    try:
        print("DEBUG: Creando vectorstore...")
        vectorstore = crear_vectorstore(discursos)
        print("DEBUG: Vectorstore creado exitosamente")
    except Exception as e:
        print(f"DEBUG: Error creando vectorstore: {str(e)}")
        st.error(f"Error al crear vectorstore: {str(e)}")
        return None, None

    # Configurar RAG
    try:
        print("DEBUG: Configurando RAG...")
        sistema_rag = configurar_rag(vectorstore)
        print("DEBUG: RAG configurado exitosamente")
    except Exception as e:
        print(f"DEBUG: Error configurando RAG: {str(e)}")
        st.error(f"Error al configurar RAG: {str(e)}")
        return None, None

    return sistema_rag, discursos

sistema_rag, discursos = inicializar_sistema()
print("DEBUG: Sistema RAG inicializado correctamente")

# Sidebar con estadísticas y ejemplos
with st.sidebar:
    st.header("📊 Estadísticas del Corpus")

    if discursos:
        num_discursos = len(discursos)

        st.metric("Número de discursos", num_discursos)

    st.header("💡 Ejemplos de Preguntas")
    ejemplos = [
        "¿Qué dice Milei sobre la economía?",
        "¿Cuáles son las prioridades del gobierno?",
        "¿Qué medidas propone para la inflación?",
        "¿Cómo describe la situación actual del país?"
    ]

    for ejemplo in ejemplos:
        if st.button(ejemplo, key=ejemplo):
            st.session_state.pregunta_input = ejemplo

    st.markdown("### 🧭 Navegación")
    st.markdown("[🔍 Ir a Consulta](#consulta)")

if sistema_rag is None:
    st.stop()


# Interfaz de consulta
st.markdown('<a id="consulta"></a>', unsafe_allow_html=True)
st.header("💬 Realizar Consulta")

pregunta = st.text_input("Ingresa tu pregunta sobre los discursos de Javier Milei:", placeholder="¿Qué dice Milei sobre la economía?", value=st.session_state.get('pregunta_input', ''))
submitted = st.button("🔍 Consultar", type="primary")

if submitted and pregunta.strip():
    with st.spinner("🔄 Procesando consulta..."):
        try:
            print(f"DEBUG: Llamando a consultar_rag con pregunta: {pregunta}")
            result = consultar_rag(sistema_rag, pregunta)
            print(f"DEBUG: Resultado recibido: {result}")
            print(f"DEBUG: Verificando resultado: keys = {list(result.keys())}")
            answer = result["result"]
            print(f"DEBUG: Answer extraído: {answer}")
            sources_raw = result["source_documents"]
            print(f"DEBUG: Sources raw: {len(sources_raw)} documentos")
            sources = []
            for doc in sources_raw:
                sources.append({
                    "titulo": doc.metadata.get("titulo", "Sin título"),
                    "fecha_publicacion": doc.metadata.get("fecha_publicacion", "Sin fecha"),
                    "url": doc.metadata.get("url", "Sin URL")
                })
            formatted_sources = "\n".join([f"- **{s['titulo']}** ({s['fecha_publicacion']}) - [{s['url']}]({s['url']})" for s in sources])
            content = answer + "\n\n📚 **Fuentes Consultadas:**\n" + formatted_sources
            print(f"DEBUG: Content final: {content[:200]}...")
        except Exception as e:
            print(f"DEBUG: Error en consulta: {str(e)}")
            st.error(f"Error al procesar la consulta: {str(e)}")
            st.stop()

    st.success("✅ Consulta procesada exitosamente.")

    with st.chat_message("user"):
        st.write(pregunta)
    with st.chat_message("assistant"):
        st.write(content)

    # Limpiar el input después de consultar
    if 'pregunta_input' in st.session_state:
        del st.session_state['pregunta_input']
elif submitted and not pregunta.strip():
    st.warning("Por favor, ingresa una pregunta.")

 

with st.expander("Información del Sistema"):

    st.markdown("**Nota:** Este sistema utiliza únicamente información de discursos oficiales disponibles en el sitio web de Casa Rosada.")