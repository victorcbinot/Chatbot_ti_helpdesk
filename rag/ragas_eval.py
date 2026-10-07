"""
ragas_eval.py — Avaliação de qualidade do RAG com RAGAS.

Roda um conjunto de perguntas de teste sobre as duas configurações de
chunking (ver splitter.py: "pequeno_512" e "grande_1024"), calcula
faithfulness e answer_relevancy para cada uma, e gera uma tabela
comparativa — o requisito central do CKP02 ("a escolha final é
justificada pelos dados?").

IMPORTANTE: o resultado é sempre salvo em ragas_resultado.md na raiz do
projeto — e esse arquivo deve ser commitado no repositório (não colocar
no .gitignore). Gerar a métrica mas não guardar a evidência foi
exatamente o motivo da perda de pontos no diferencial do CKP01.

Execução:
    python -m rag.ragas_eval
"""

from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, faithfulness

from rag.embeddings_store import build_embeddings, carregar_vectorstore
from rag.rag_chain import build_llm, build_rag_chain
from rag.splitter import CONFIGURACOES_CHUNKING

# >= 5 perguntas de teste (requisito obrigatório do CKP02), cobrindo as
# categorias do domínio representadas nos documentos da base: hardware,
# rede, software e acesso/segurança.
PERGUNTAS_TESTE = [
    "Meu computador não liga, o que eu devo verificar primeiro?",
    "Como posso diagnosticar um problema de conectividade de rede usando ping e ipconfig?",
    "O que é autenticação multifator resistente a phishing?",
    "Quais são boas práticas para proteger o acesso remoto a sistemas corporativos?",
    "Como solucionar problemas de uma aplicação Java que está travando?",
    "Se o notebook não liga nem com o cabo de energia conectado, o que pode ser o problema?",
]


def _coletar_respostas_e_contextos(
    perguntas: list[str], config_nome: str, k: int = 4
) -> tuple[list[str], list[list[str]]]:
    """
    Para cada pergunta, roda o retriever + generate e coleta (resposta,
    contextos usados) — entrada necessária para calcular o RAGAS.
    """
    vectorstore = carregar_vectorstore(config_nome)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    chain = build_rag_chain(config_nome, k=k)

    respostas: list[str] = []
    contextos: list[list[str]] = []

    for pergunta in perguntas:
        documentos_recuperados = retriever.invoke(pergunta)
        contextos.append([d.page_content for d in documentos_recuperados])

        resposta = chain.invoke(pergunta)
        respostas.append(resposta)

    return respostas, contextos


def avaliar_configuracao(
    config_nome: str, perguntas: list[str] = PERGUNTAS_TESTE
) -> dict:
    """
    Avalia uma configuração de chunking com RAGAS (faithfulness +
    answer_relevancy), retornando as métricas médias e o detalhe por
    pergunta (como DataFrame do pandas).
    """
    print(f"\nAvaliando configuração '{config_nome}'...")
    respostas, contextos = _coletar_respostas_e_contextos(perguntas, config_nome)

    dataset = Dataset.from_dict(
        {
            "question": perguntas,
            "answer": respostas,
            "contexts": contextos,
        }
    )

    llm_ragas = LangchainLLMWrapper(build_llm(temperature=0.0))
    embeddings_ragas = LangchainEmbeddingsWrapper(build_embeddings())

    resultado = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm_ragas,
        embeddings=embeddings_ragas,
    )

    df = resultado.to_pandas()
    return {
        "config_nome": config_nome,
        "faithfulness_medio": df["faithfulness"].mean(),
        "answer_relevancy_medio": df["answer_relevancy"].mean(),
        "detalhes": df,
    }


def comparar_configuracoes(
    configuracoes: list[str] | None = None,
    caminho_saida: str = "ragas_resultado.md",
) -> list[dict]:
    """
    Avalia TODAS as configurações de chunking definidas em splitter.py,
    imprime e salva uma tabela comparativa em Markdown.
    """
    configuracoes = configuracoes or list(CONFIGURACOES_CHUNKING.keys())
    resultados = [avaliar_configuracao(c) for c in configuracoes]

    linhas = [
        "| Configuração | chunk_size | Faithfulness médio | Answer Relevancy médio |",
        "|---|---|---|---|",
    ]
    for r in resultados:
        chunk_size = CONFIGURACOES_CHUNKING[r["config_nome"]]["chunk_size"]
        linhas.append(
            f"| {r['config_nome']} | {chunk_size} | "
            f"{r['faithfulness_medio']:.3f} | {r['answer_relevancy_medio']:.3f} |"
        )
    tabela_md = "\n".join(linhas)
    print("\n" + tabela_md)

    # Proteção: se alguma métrica vier NaN (o RAGAS não conseguiu parsear a
    # resposta do modelo em alguma pergunta), avisa claramente em vez de
    # escolher a "melhor configuração" de forma arbitrária/silenciosa.
    resultados_validos = [
        r for r in resultados if r["faithfulness_medio"] == r["faithfulness_medio"]  # filtra NaN
    ]
    if len(resultados_validos) < len(resultados):
        print(
            "\n⚠️  Aviso: alguma configuração teve faithfulness = NaN "
            "(o RAGAS não conseguiu interpretar a resposta do modelo em "
            "pelo menos uma pergunta). Considere reduzir a temperatura, "
            "rodar de novo, ou simplificar as perguntas de teste."
        )
    if not resultados_validos:
        print("\n❌ Nenhuma configuração teve resultado válido — não é possível escolher a melhor.")
        return resultados

    melhor = max(resultados_validos, key=lambda r: r["faithfulness_medio"])

    with open(caminho_saida, "w", encoding="utf-8") as f:
        f.write("# Comparação de Chunking — RAGAS\n\n")
        f.write(
            "Avaliação feita com as métricas `faithfulness` e "
            "`answer_relevancy` do RAGAS, sobre as perguntas de teste "
            "definidas em `rag/ragas_eval.py`.\n\n"
        )
        f.write(tabela_md + "\n\n")
        f.write(
            f"**Configuração escolhida:** `{melhor['config_nome']}` "
            f"(maior faithfulness médio: {melhor['faithfulness_medio']:.3f})\n\n"
        )
        for r in resultados:
            f.write(f"\n## Detalhes por pergunta — {r['config_nome']}\n\n")
            colunas = [
                c
                for c in ["question", "faithfulness", "answer_relevancy"]
                if c in r["detalhes"].columns
            ]
            f.write(r["detalhes"][colunas].to_markdown(index=False))
            f.write("\n")

    print(f"\nResultado salvo em: {caminho_saida}")
    print(
        f"Melhor configuração (faithfulness): {melhor['config_nome']} "
        f"({melhor['faithfulness_medio']:.3f})"
    )

    return resultados


if __name__ == "__main__":
    comparar_configuracoes()