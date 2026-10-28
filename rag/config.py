"""
config.py — Configuração vencedora do RAG (única fonte da verdade).

Centraliza os parâmetros escolhidos na avaliação RAGAS da CP2
(ver ragas_resultado.md na raiz do projeto), para que rag_chain.py,
rag/main.py, agent/tools.py e as interfaces usem SEMPRE os mesmos
valores, sem duplicar constantes em outros arquivos.

Vencedora: ``pequeno_512`` — maior faithfulness médio (0.847 vs 0.735
da grande_1024). Os parâmetros de chunking em si (chunk_size/chunk_overlap)
ficam em rag/splitter.py:CONFIGURACOES_CHUNKING; aqui fica apenas QUAL
configuração venceu e quantos trechos seguem para o LLM.

Reranking e filtro de metadata NÃO são fixados aqui: continuam
opcionais (env RERANKER_ATIVO / parâmetro ``rerank``; parâmetro ``filtro``
no formato ``where`` do ChromaDB), preservando o comportamento da
configuração vencedora — que foi medida SEM filtro (reranking conforme
o env RERANKER_ATIVO).
"""

# Nome da configuração de chunking vencedora no RAGAS (ver ragas_resultado.md).
CONFIG_PADRAO_RAG = "pequeno_512"

# Quantos trechos (top_k) seguem para o modelo de geração após a recuperação.
K_TRECHOS = 4
