"""
embeddings_store.py — Etapas "embed" e "store" do pipeline RAG.

Gera embeddings dos chunks com nomic-embed-text (via Ollama Cloud) e
armazena no ChromaDB local — uma coleção por configuração de chunking,
para permitir comparar as duas estratégias (512 vs 1024) de forma isolada.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

load_dotenv()

# Pasta onde o ChromaDB persiste os dados em disco (na raiz do projeto).
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"

# Nome base do domínio, usado para nomear as coleções do ChromaDB.
DOMINIO = "ti_helpdesk"


def _get_api_key() -> str:
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OLLAMA_API_KEY não encontrada. Copie .env.example para .env "
            "e preencha sua chave da Ollama Cloud."
        )
    return api_key


def build_embeddings() -> OllamaEmbeddings:
    """
    Instancia o modelo de embeddings nomic-embed-text via Ollama Cloud.
    Único modelo de embedding aprovado pelo enunciado do CKP02.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
    return OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=base_url,
        client_kwargs={"headers": {"Authorization": f"Bearer {_get_api_key()}"}},
    )


def nome_colecao(config_nome: str) -> str:
    """Nome da coleção do ChromaDB para uma dada configuração de chunking."""
    return f"{DOMINIO}_{config_nome}"


def criar_vectorstore(
    chunks: list[Document],
    config_nome: str,
    persist_dir: Path = CHROMA_DIR,
    embeddings: OllamaEmbeddings | None = None,
) -> Chroma:
    """
    Gera os embeddings dos chunks informados e os armazena em uma coleção
    do ChromaDB (uma coleção por configuração de chunking).

    O parâmetro `embeddings` existe principalmente para permitir testes
    automatizados com um modelo de embeddings substituto, sem depender da
    Ollama Cloud de verdade.
    """
    embeddings = embeddings or build_embeddings()
    colecao = nome_colecao(config_nome)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=colecao,
        persist_directory=str(persist_dir),
    )
    print(f"  [{colecao}] {len(chunks)} chunks armazenados em {persist_dir}")
    return vectorstore


def carregar_vectorstore(
    config_nome: str,
    persist_dir: Path = CHROMA_DIR,
    embeddings: OllamaEmbeddings | None = None,
) -> Chroma:
    """Carrega uma coleção já existente do ChromaDB, sem gerar embeddings de novo."""
    embeddings = embeddings or build_embeddings()
    colecao = nome_colecao(config_nome)
    return Chroma(
        collection_name=colecao,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


if __name__ == "__main__":
    from app.loader import carregar_documentos
    from app.splitter import dividir_com_todas_configuracoes

    documentos = carregar_documentos()
    print()
    configuracoes = dividir_com_todas_configuracoes(documentos)

    print("\nGerando embeddings e armazenando no ChromaDB...")
    for nome, chunks in configuracoes.items():
        criar_vectorstore(chunks, nome)

    print("\nPronto. Para testar uma busca, use carregar_vectorstore() e .similarity_search().")