import json
import os

def agregar_transcripcion():
    # Ruta al archivo JSON
    archivo_json = 'discursos.json'

    # Verificar si el archivo existe
    if not os.path.exists(archivo_json):
        print(f"El archivo {archivo_json} no existe.")
        return

    # Cargar el JSON existente
    try:
        with open(archivo_json, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
    except json.JSONDecodeError:
        print("Error al cargar el archivo JSON.")
        return

    # Pedir datos al usuario
    titulo = input("Ingrese el título de la transcripción: ")
    contenido = input("Ingrese el contenido de la transcripción: ")
    fecha = input("Ingrese la fecha (formato YYYY-MM-DD): ")
    url = input("Ingrese la URL (opcional, presione Enter para omitir): ").strip()

    # Crear el nuevo documento
    nuevo_documento = {
        "titulo": titulo,
        "contenido": contenido,
        "fecha": fecha
    }
    if url:
        nuevo_documento["url"] = url

    # Agregar al corpus (asumiendo que es una lista)
    if isinstance(corpus, list):
        corpus.append(nuevo_documento)
    else:
        print("El corpus no es una lista. No se puede agregar.")
        return

    # Guardar el JSON actualizado
    try:
        with open(archivo_json, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, ensure_ascii=False, indent=4)
        print("Transcripción agregada exitosamente.")
    except Exception as e:
        print(f"Error al guardar el archivo: {e}")

if __name__ == "__main__":
    agregar_transcripcion()