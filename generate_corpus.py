import json
from mvp_rag import ScraperDiscursosCasaRosada

def main():
    scraper = ScraperDiscursosCasaRosada()
    discursos = scraper.procesar_discursos()

    with open('discursos.json', 'w', encoding='utf-8') as f:
        json.dump(discursos, f, ensure_ascii=False, indent=4)

    print(f"Se han guardado {len(discursos)} discursos en discursos.json")

if __name__ == "__main__":
    main()