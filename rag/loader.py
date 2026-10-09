"""
loader.py — Etapa "load" do pipeline RAG.

Carrega todos os documentos PDF da pasta docs/ (a base de conhecimento real
do domínio de TI) e retorna uma lista de Document do LangChain, um por
página, já com metadata de origem (nome do arquivo e número da página) e de
classificação (categoria, fornecedor e tipo de documento), para permitir
Metadata Filtering na busca (ver rag/metadata.py).
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from rag.metadata import METADADOS_POR_ARQUIVO, metadados_do_arquivo

# docs/ fica na raiz do projeto, um nível acima da pasta app/
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


def carregar_documentos(docs_dir: Path = DOCS_DIR) -> list[Document]:
    """
    Carrega todos os arquivos .pdf da pasta docs/.

    Cada página de cada PDF vira um Document separado, com metadata:
      - source: nome do arquivo PDF de origem (usado para citar a fonte)
      - page: número da página dentro do PDF
      - categoria, fornecedor, tipo_documento: classificação registrada em
        rag/metadata.py, usada para Metadata Filtering

    Levanta FileNotFoundError se a pasta não existir ou estiver vazia,
    para falhar cedo com uma mensagem clara em vez de seguir com uma
    base de conhecimento vazia.
    """
    if not docs_dir.exists():
        raise FileNotFoundError(
            f"Pasta de documentos não encontrada: {docs_dir}. "
            "Crie a pasta docs/ e coloque os PDFs da base de conhecimento."
        )

    arquivos_pdf = sorted(docs_dir.glob("*.pdf"))
    if not arquivos_pdf:
        raise FileNotFoundError(
            f"Nenhum arquivo .pdf encontrado em {docs_dir}. "
            "Adicione os documentos reais da base de conhecimento antes de continuar."
        )

    documentos: list[Document] = []
    for caminho in arquivos_pdf:
        loader = PyPDFLoader(str(caminho))
        paginas = loader.load()

        metadados = metadados_do_arquivo(caminho.name)
        if caminho.name not in METADADOS_POR_ARQUIVO:
            print(
                f"  [AVISO] Sem metadados registrados para {caminho.name}; "
                f"categoria='{metadados['categoria']}'. "
                "Registre-o em rag/metadata.py para poder filtrá-lo."
            )

        for pagina in paginas:
            # Garante que o metadata 'source' seja só o nome do arquivo (não o
            # caminho completo), para citar a fonte de forma limpa nas respostas.
            pagina.metadata["source"] = caminho.name
            # Classificação (categoria, fornecedor, tipo_documento) usada no
            # Metadata Filtering. Preservada automaticamente nos chunks.
            pagina.metadata.update(metadados)

        documentos.extend(paginas)
        print(f"  [ok] {caminho.name}: {len(paginas)} página(s)")

    print(
        f"\nTotal: {len(documentos)} páginas carregadas de "
        f"{len(arquivos_pdf)} documento(s) PDF."
    )
    return documentos


def listar_documentos_fonte(docs_dir: Path = DOCS_DIR) -> list[str]:
    """Retorna só os nomes dos arquivos PDF na pasta docs/, sem carregar o conteúdo."""
    return [p.name for p in sorted(docs_dir.glob("*.pdf"))]


if __name__ == "__main__":
    documentos = carregar_documentos()
    print("\nPrévia das 3 primeiras páginas carregadas:\n")
    for doc in documentos[:3]:
        fonte = doc.metadata.get("source")
        pagina = doc.metadata.get("page")
        categoria = doc.metadata.get("categoria")
        print(f"--- {fonte} (página {pagina}, categoria {categoria}) ---")
        print(doc.page_content[:200].strip().replace("\n", " "))
        print()