# Comparação de Chunking - RAGAS

Avaliação feita com as métricas `faithfulness` e `answer_relevancy` do RAGAS, sobre as perguntas de teste definidas em `rag/ragas_eval.py`.

| Configuração | chunk_size | Faithfulness médio | Answer Relevancy médio |
|---|---|---|---|
| pequeno_512 | 512 | 0.847 | 0.406 |
| grande_1024 | 1024 | 0.735 | 0.152 |

**Configuração escolhida:** `pequeno_512` (maior faithfulness médio: 0.847)


## Detalhes por pergunta - pequeno_512

|   faithfulness |   answer_relevancy |
|---------------:|-------------------:|
|       1        |           0        |
|       0.9      |           0.870587 |
|       0.666667 |           0.846347 |
|       0.714286 |           0.716645 |
|       0.8      |           0        |
|       1        |           0        |

## Detalhes por pergunta - grande_1024

|   faithfulness |   answer_relevancy |
|---------------:|-------------------:|
|       1        |           0        |
|       1        |           0        |
|       0.909091 |           0.913141 |
|       0        |           0        |
|       1        |           0        |
|       0.5      |           0        |
