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

from rag.config import CONFIG_PADRAO_RAG, K_TRECHOS
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


def _recuperar_trechos(
    vectorstore,
    consulta: str,
    k: int,
    filtro: dict | None,
    usar_rerank: bool,
) -> list[Document]:
    """
    Etapa retrieve compartilhada por TODOS os caminhos de busca
    (buscar_chunks, build_rag_chain e buscar_com_fontes) — a lógica de
    recuperação existe UMA vez. Com reranking, recupera ``fetch_k``
    candidatos no ChromaDB e reordena via cross-encoder; sem, devolve os
    ``k`` melhores diretos. Em falha do cross-encoder, degrada para a
    ordem original (mesmo comportamento anterior da chain).
    """
    n_candidatos = max(fetch_k(), k) if usar_rerank else k
    candidatos = vectorstore.similarity_search(
        consulta, k=n_candidatos, filter=filtro
    )
    if not usar_rerank:
        return candidatos[:k]
    try:
        return rerank_documentos(consulta, candidatos, top_k=k)
    except Exception as exc:  # noqa: BLE001 — degrada com aviso
        print(f"[AVISO] Reranking falhou, usando ordem original: {exc}")
        return candidatos[:k]


def _resolver_rerank(rerank: bool | None) -> bool:
    """None usa o padrão do env (RERANKER_ATIVO); caso contrário, usa o valor."""
    return reranker_ativo() if rerank is None else rerank


def _chain_geracao(llm: ChatOllama):
    """Etapa generate compartilhada: RAG_PROMPT | llm | StrOutputParser()."""
    return RAG_PROMPT | llm | StrOutputParser()


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
    config_nome: str = CONFIG_PADRAO_RAG,
    k: int = K_TRECHOS,
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
    usar_rerank = _resolver_rerank(rerank)
    vectorstore = carregar_vectorstore(config_nome)
    return _recuperar_trechos(vectorstore, consulta, k, filtro, usar_rerank)


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
    usar_rerank = _resolver_rerank(rerank)
    vectorstore = carregar_vectorstore(config_nome)
    llm = llm or build_llm()

    def _recuperar_rerank_formatar(pergunta: str) -> str:
        trechos = _recuperar_trechos(
            vectorstore, pergunta, k, filtro, usar_rerank
        )
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
    config_nome: str = CONFIG_PADRAO_RAG,
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

    Esta é a função exposta como @tool no CKP03 (Agente) — a assinatura é
    mantida simples (str -> str) por padrão. Quando as fontes também são
    necessárias, use buscar_com_fontes(), que roda o MESMO pipeline
    (uma única recuperação + uma única geração).
    """
    resposta, _trechos = buscar_com_fontes(
        consulta, config_nome=config_nome, filtro=filtro, rerank=rerank
    )
    return resposta


def buscar_com_fontes(
    consulta: str,
    config_nome: str = CONFIG_PADRAO_RAG,
    filtro: dict | None = None,
    rerank: bool | None = None,
    llm: ChatOllama | None = None,
    k: int = K_TRECHOS,
) -> tuple[str, list[Document]]:
    """
    Executa o pipeline RAG completo UMA vez (retrieve -> generate) e devolve
    (resposta, trechos): a resposta gerada e os Documents que a fundamentaram,
    com os metadados originais preservados (source, page, categoria,
    fornecedor, tipo_documento).

    Reutiliza as MESMAS etapas do pipeline existente — recuperação via
    _recuperar_trechos ( ChromaDB + filtro de metadata opcional + reranking
    opcional do cross-encoder) e geração via _chain_geracao
    (RAG_PROMPT | llm gemma4:cloud | StrOutputParser) — sem duplicar a
    lógica e SEM rodar a busca/reranking duas vezes.

    O parâmetro `llm` existe para testes automatizados com um modelo
    substituto (mesmo contrato de build_rag_chain); None usa build_llm().
    """
    usar_rerank = _resolver_rerank(rerank)
    vectorstore = carregar_vectorstore(config_nome)
    trechos = _recuperar_trechos(vectorstore, consulta, k, filtro, usar_rerank)
    contexto = _formatar_contexto(trechos) if trechos else ""
    resposta = _chain_geracao(llm or build_llm()).invoke(
        {"contexto": contexto, "pergunta": consulta}
    )
    return resposta, trechos

if __name__ == "__main__":
    pergunta_teste = "Meu computador não liga, o que eu faço?"
    print(f"Pergunta: {pergunta_teste}\n")
    resposta = buscar(pergunta_teste)
    print(f"Resposta:\n{resposta}")