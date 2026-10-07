"""
splitter.py — Etapa "split" do pipeline RAG.

Divide os documentos carregados em chunks menores, usando o
RecursiveCharacterTextSplitter (técnica da Aula 06), com os separadores
recomendados no enunciado.

Define as 2 configurações de chunk_size que serão comparadas com RAGAS
(requisito obrigatório do CKP02): uma configuração menor (chunks mais
granulares, mais precisos mas com menos contexto por chunk) e uma maior
(chunks com mais contexto, mas potencialmente menos precisos na busca).
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Separadores recomendados no enunciado (Aula 06): tenta quebrar primeiro em
# parágrafos, depois linhas, depois frases, depois espaços, só then caractere a caractere.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# As 2 configurações de chunking que serão comparadas com RAGAS.
# chunk_overlap definido como 15% do chunk_size (dentro da faixa 10-15% pedida).
CONFIGURACOES_CHUNKING = {
    "pequeno_512": {"chunk_size": 512, "chunk_overlap": 77},
    "grande_1024": {"chunk_size": 1024, "chunk_overlap": 154},
}


def dividir_documentos(
    documentos: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Divide uma lista de Document em chunks, com o chunk_size/overlap dados."""
    splitter = RecursiveCharacterTextSplitter(
        separators=SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documentos)


def dividir_com_todas_configuracoes(
    documentos: list[Document],
) -> dict[str, list[Document]]:
    """
    Aplica TODAS as configurações definidas em CONFIGURACOES_CHUNKING sobre
    os mesmos documentos, retornando um dicionário
    {nome_da_configuracao: lista_de_chunks}.

    Isso é o que permite comparar as 2 estratégias de chunking lado a lado
    mais adiante, quando calcularmos o RAGAS para cada uma.
    """
    resultado: dict[str, list[Document]] = {}
    for nome, config in CONFIGURACOES_CHUNKING.items():
        chunks = dividir_documentos(
            documentos,
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
        )
        resultado[nome] = chunks
        print(
            f"  [{nome}] chunk_size={config['chunk_size']}, "
            f"chunk_overlap={config['chunk_overlap']} -> {len(chunks)} chunks"
        )
    return resultado


if __name__ == "__main__":
    from app.loader import carregar_documentos

    documentos = carregar_documentos()
    print()
    resultado = dividir_com_todas_configuracoes(documentos)

    for nome, chunks in resultado.items():
        print(f"\n--- Exemplo de chunk em '{nome}' ---")
        print(chunks[0].page_content[:300].strip())
        print(f"(fonte: {chunks[0].metadata.get('source')})")