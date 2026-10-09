"""
rag_chain.py — Etapas "retrieve" e "generate" do pipeline RAG.

Monta a chain RAG completa como um ÚNICO objeto Runnable
(retriever | prompt | llm | parser), separado da interface — essa é a
lição aplicada do feedback do CKP01 ("mantenha a chain como um objeto
Runnable único, separado da interface, para poder encadear com o
recuperador").

O diferencial de reranking (rag/reranker.py) entra entre o retrieve e o
generate: o ChromaDB recupera mais candidatos (fetch_k) e um cross-encoder
reordena os top_k finais por relevância antes de montar o prompt. O
reranker é opcional e pode ser desativado (env RERANKER_ATIVO=false ou
rerank=False), restaurando o comportamento original.

A função buscar(consulta) é a interface de alto nível desse módulo: ela é
a que será reaproveitada como @tool no CKP03 (Agente), conforme avisado no
enunciado — por isso sua assinatura é simples (recebe uma string, devolve
uma string).
"""

import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama

from rag.embeddings_store import carregar_vectorstore
from rag.reranker import fetch_k, rerank_documentos, reranker_ativo

load_dotenv()

MODELO = "gemma4:cloud"

# System prompt com XML tagging (mesma técnica usada no CKP01), adaptado
# para o cenário de RAG: a persona agora responde SOMENTE com base no
# contexto recuperado, nunca "de cabeça".
SYSTEM_PROMPT_RAG = """
<persona>
Você é o DocMind, um assistente de suporte técnico de TI que responde
exclusivamente com base em documentos técnicos reais fornecidos como
contexto — não em conhecimento geral memorizado.
</persona>

<regras>
1. Responda SOMENTE com base no contexto fornecido abaixo. Não use
   conhecimento prévio que não esteja no contexto.
2. Sempre cite o nome do documento de origem (campo "Fonte") que embasou
   a resposta, ao final da resposta.
3. Se o contexto não tiver informação suficiente para responder com
   segurança, diga claramente que não encontrou essa informação na base
   de conhecimento, em vez de inventar uma resposta.
</regras>

<restricoes>
- Nunca invente informações que não estejam explicitamente no contexto.
- Nunca omita a citação da fonte quando usar uma informação do contexto.
</restricoes>
""".strip()

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT_RAG),
        ("human", "Contexto:\n{contexto}\n\nPergunta: {pergunta}"),
    ]
)


def _get_api_key() -> str:
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OLLAMA_API_KEY não encontrada. Copie .env.example para .env "
            "e preencha sua chave da Ollama Cloud."
        )
    return api_key


def build_llm(temperature: float = 0.0) -> ChatOllama:
    """Instancia o ChatOllama (gemma4:cloud). temperature=0 para respostas fundamentadas."""
    base_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
    return ChatOllama(
        model=MODELO,
        base_url=base_url,
        temperature=temperature,
        client_kwargs={"headers": {"Authorization": f"Bearer {_get_api_key()}"}},
    )


def _formatar_contexto(documentos) -> str:
    """
    Formata os chunks recuperados pelo retriever em um único texto,
    citando claramente a fonte (e página, se disponível) de cada trecho.
    """
    partes = []
    for doc in documentos:
        fonte = doc.metadata.get("source", "desconhecido")
        pagina = doc.metadata.get("page")
        cabecalho = f"[Fonte: {fonte}" + (
            f", página {pagina}]" if pagina is not None else "]"
        )
        partes.append(f"{cabecalho}\n{doc.page_content}")
    return "\n\n---\n\n".join(partes)


def buscar_chunks(
    consulta: str,
    config_nome: str = "pequeno_512",
    k: int = 4,
    filtro: dict | None = None,
    rerank: bool | None = None,
) -> list[Document]:
    """
    Recupera os trechos (Documents) que seriam usados na resposta de uma
    consulta, opcionalmente aplicando um filtro de metadata e/ou reranking.

    O `filtro` é o dict no formato ``where`` do ChromaDB (ver
    rag.metadata.construir_filtro). Quando None, a busca acontece sem filtro
    — exatamente como antes do diferencial de Metadata Filtering.

    `rerank`: True ativa o cross-encoder, False desativa, None usa
    RERANKER_ATIVO (env). Com reranking ativo, o ChromaDB recupera
    ``fetch_k`` candidatos (> k) e o reranker reordena os ``k`` melhores.

    Retorna os Documents já com metadata (source, page, categoria,
    fornecedor, tipo_documento) preservado.
    """
    usar_rerank = reranker_ativo() if rerank is None else rerank
    vectorstore = carregar_vectorstore(config_nome)

    n_candidatos = max(fetch_k(), k) if usar_rerank else k
    candidatos = vectorstore.similarity_search(
        consulta, k=n_candidatos, filter=filtro
    )
    if not usar_rerank:
        return candidatos[:k]
    return rerank_documentos(consulta, candidatos, top_k=k)


def build_rag_chain(
    config_nome: str,
    k: int = 4,
    llm: ChatOllama | None = None,
    filtro: dict | None = None,
    rerank: bool | None = None,
):
    """
    Monta a chain RAG completa (retrieve + generate) como um único
    Runnable, para uma dada configuração de chunking (ex: "pequeno_512"
    ou "grande_1024" — ver splitter.py).

    O parâmetro `llm` existe para permitir testes automatizados com um
    modelo substituto, sem depender da Ollama Cloud de verdade.

    O parâmetro `filtro` (formato ``where`` do ChromaDB) restringe a
    recuperação a determinados metadados. Se for None, o comportamento é o
    mesmo de antes (busca sem filtro).

    `rerank`: True ativa o reranking (cross-encoder) entre o retrieve e o
    generate; False desativa; None usa RERANKER_ATIVO (env). Com o
    reranking ativo o ChromaDB recupera ``fetch_k`` candidatos e apenas os
    ``k`` melhores depois do rerank vão para o LLM.
    """
    usar_rerank = reranker_ativo() if rerank is None else rerank
    vectorstore = carregar_vectorstore(config_nome)
    n_candidatos = max(fetch_k(), k) if usar_rerank else k
    search_kwargs: dict = {"k": n_candidatos}
    if filtro:
        search_kwargs["filter"] = filtro
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    llm = llm or build_llm()

    def _recuperar_rerank_formatar(pergunta: str) -> str:
        trechos = retriever.invoke(pergunta)
        if not trechos:
            return ""
        if usar_rerank:
            try:
                trechos = rerank_documentos(pergunta, trechos, top_k=k)
            except Exception as exc:  # noqa: BLE001 — degrada com aviso
                print(f"[AVISO] Reranking falhou, usando ordem original: {exc}")
                trechos = trechos[:k]
        else:
            trechos = trechos[:k]
        return _formatar_contexto(trechos)

    # A chain inteira é um único Runnable: recebe a pergunta (string),
    # recupera o contexto (com reranking opcional), gera o prompt, chama o
    # modelo, e faz o parsing final — tudo encadeado com o operador `|`.
    chain = (
        {
            "contexto": RunnableLambda(_recuperar_rerank_formatar),
            "pergunta": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def buscar(
    consulta: str,
    config_nome: str = "pequeno_512",
    filtro: dict | None = None,
    rerank: bool | None = None,
) -> str:
    """
    Função de busca de alto nível: recebe uma pergunta em texto simples e
    devolve a resposta gerada pelo RAG, já citando a fonte usada.

    O parâmetro opcional `filtro` (formato ``where`` do ChromaDB, ver
    rag.metadata.construir_filtro) restringe a busca a metadados
    específicos. Sem ele, a busca ocorre sem filtro — comportamento original.

    O parâmetro opcional `rerank` ativa (True) ou desativa (False) o
    reranking; None usa RERANKER_ATIVO (env).

    Esta é a função que será reaproveitada como @tool no CKP03 (Agente) —
    por isso a assinatura é mantida simples (str -> str) por padrão.
    """
    chain = build_rag_chain(config_nome, filtro=filtro, rerank=rerank)
    return chain.invoke(consulta)


def buscar_com_fontes(
    consulta: str,
    config_nome: str = "pequeno_512",
    filtro: dict | None = None,
    rerank: bool | None = None,
) -> tuple[str, list[Document]]:
    """
    Responde a uma consulta e devolve também os trechos (Documents, com
    metadata: source, page, categoria, ...) que fundamentaram a resposta.

    Reusa buscar() (resposta) e buscar_chunks() (fontes) — ambos passam pelo
    mesmo caminho retrieve (+ filtro de metadata e reranking opcionais) — sem
    duplicar a lógica do pipeline. É a função usada pela interface Streamlit.
    """
    resposta = buscar(consulta, config_nome=config_nome, filtro=filtro, rerank=rerank)
    trechos = buscar_chunks(
        consulta, config_nome=config_nome, k=4, filtro=filtro, rerank=rerank
    )
    return resposta, trechos

if __name__ == "__main__":
    pergunta_teste = "Meu computador não liga, o que eu faço?"
    print(f"Pergunta: {pergunta_teste}\n")
    resposta = buscar(pergunta_teste)
    print(f"Resposta:\n{resposta}")