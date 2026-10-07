"""
rag_chain.py — Etapas "retrieve" e "generate" do pipeline RAG.

Monta a chain RAG completa como um ÚNICO objeto Runnable
(retriever | prompt | llm | parser), separado da interface — essa é a
lição aplicada do feedback do CKP01 ("mantenha a chain como um objeto
Runnable único, separado da interface, para poder encadear com o
recuperador").

A função buscar(consulta) é a interface de alto nível desse módulo: ela é
a que será reaproveitada como @tool no CKP03 (Agente), conforme avisado no
enunciado — por isso sua assinatura é simples (recebe uma string, devolve
uma string).
"""

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from rag.embeddings_store import carregar_vectorstore

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


def build_rag_chain(config_nome: str, k: int = 4, llm: ChatOllama | None = None):
    """
    Monta a chain RAG completa (retrieve + generate) como um único
    Runnable, para uma dada configuração de chunking (ex: "pequeno_512"
    ou "grande_1024" — ver splitter.py).

    O parâmetro `llm` existe para permitir testes automatizados com um
    modelo substituto, sem depender da Ollama Cloud de verdade.
    """
    vectorstore = carregar_vectorstore(config_nome)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = llm or build_llm()

    # A chain inteira é um único Runnable: recebe a pergunta (string),
    # recupera o contexto, gera o prompt, chama o modelo, e faz o parsing
    # final da saída — tudo encadeado com o operador `|`.
    chain = (
        {
            "contexto": retriever | _formatar_contexto,
            "pergunta": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def buscar(consulta: str, config_nome: str = "grande_1024") -> str:
    """
    Função de busca de alto nível: recebe uma pergunta em texto simples e
    devolve a resposta gerada pelo RAG, já citando a fonte usada.

    Esta é a função que será reaproveitada como @tool no CKP03 (Agente) —
    por isso a assinatura é mantida simples (str -> str), conforme pedido
    no enunciado do CKP02.
    """
    chain = build_rag_chain(config_nome)
    return chain.invoke(consulta)


if __name__ == "__main__":
    pergunta_teste = "Meu computador não liga, o que eu faço?"
    print(f"Pergunta: {pergunta_teste}\n")
    resposta = buscar(pergunta_teste)
    print(f"Resposta:\n{resposta}")