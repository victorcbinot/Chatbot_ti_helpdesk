"""
reranker.py — Diferencial de Reranking (CP2).

Adiciona uma etapa de reranking ao pipeline RAG usando um cross-encoder
(cross-encoder/ms-marco-MiniLM-L-6-v2, via sentence-transformers).

Fluxo com o diferencial ativo:
    1. ChromaDB (primeira etapa) recupera FETCH_K candidatos — mais do que o
       final;
    2. o cross-encoder pontua a relação de cada candidato com a pergunta e
       reordena a lista por relevância (score descrescente);
    3. apenas os TOP_K finais seguem para o modelo de geração (gemma4:cloud).

O reranker é UMA ETAPA ADICIONAL — não substitui os componentes obrigatórios.
Pode ser desativado via env RERANKER_ATIVO=false (ou rerank=False nas
funções), restaurando o comportamento original (busca top_k direto no
ChromaDB). Ao contrário dos embeddings, o cross-encoder roda LOCALMENTE
(CPU/GPU) e é baixado da Hugging Face no primeiro uso.
"""

import os
from functools import lru_cache

from langchain_core.documents import Document

# Modelo cross-encoder de referência do enunciado. Roda localmente via
# sentence-transformers (torch) e é baixado na primeira execução.
MODELO_RERANKER_PADRAO = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def reranker_ativo() -> bool:
    """Se o reranking deve entrar no pipeline por padrão (env RERANKER_ATIVO)."""
    return os.getenv("RERANKER_ATIVO", "true").strip().lower() in (
        "1",
        "true",
        "sim",
        "yes",
    )


def fetch_k() -> int:
    """
    Quantos candidatos o ChromaDB recupera antes do reranking.
    Deve ser maior que o top_k final (env RERANKER_FETCH_K, padrão 10).
    """
    try:
        return max(1, int(os.getenv("RERANKER_FETCH_K", "10")))
    except ValueError:
        return 10


def top_k() -> int:
    """Quantos trechos finais seguem para o LLM após o reranking (env RERANKER_TOP_K)."""
    try:
        return max(1, int(os.getenv("RERANKER_TOP_K", "4")))
    except ValueError:
        return 4


@lru_cache(maxsize=1)
def _carregar_cross_encoder():
    """Carrega (e cacheia) o cross-encoder. Baixa o modelo no primeiro uso."""
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(MODELO_RERANKER_PADRAO)
    except ImportError as exc:
        raise ImportError(
            "Reranking exige 'sentence-transformers' e 'torch'. "
            "Instale com: pip install -r requirements.txt"
        ) from exc


def pontuar(consulta: str, documentos: list[Document]) -> list[float]:
    """Pontua cada trecho pela relação com a consulta (cross-encoder). Ordem preservada."""
    if not documentos:
        return []
    pares = [(consulta, doc.page_content) for doc in documentos]
    predicoes = _carregar_cross_encoder().predict(pares)
    return [float(v) for v in predicoes]


def scores_por_documento(
    consulta: str, documentos: list[Document]
) -> list[tuple[Document, float]]:
    """
    Devolve (Document, score) ordenados por score decrescente. Os Document são
    os MESMOS objetos originais — metadados, fonte e conteúdo preservados.
    """
    pontuados = list(zip(documentos, pontuar(consulta, documentos)))
    pontuados.sort(key=lambda item: item[1], reverse=True)
    return pontuados


def rerank_documentos(
    consulta: str,
    documentos: list[Document],
    top_k: int | None = None,
) -> list[Document]:
    """
    Reordena os trechos por relevância à consulta e devolve a lista (só os
    Document, na nova ordem). Metadados/fonte/conteúdo originais preservados.
    `top_k` limita o retorno; None devolve todos os candidatos reordenados.
    """
    if not documentos:
        return []
    reordenados = scores_por_documento(consulta, documentos)
    if top_k is not None and top_k > 0:
        reordenados = reordenados[:top_k]
    return [doc for doc, _ in reordenados]