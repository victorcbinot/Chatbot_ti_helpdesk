"""
tools.py — Ferramentas (@tool) do agente de triagem de TI da CP3.

4 ferramentas (>= 3 exigidas pelo enunciado), TODAS de input único
(string) — o parser ReAct (ReActSingleInputOutputParser, usado pelo
create_react_agent) passa a Action Input como UMA string, então a
assinatura das tools é propositalmente de um só parâmetro.

  1. consultar_base_conhecimento_ti   — tool RAG da CP2 (@tool). Encapsula
     ChromaDB + recuperação + reranking + geração do pipeline existente
     (rag/rag_chain.py), SEM rodar busca/reranking duas vezes: busca os
     trechos UMA vez (buscar_chunks, que já aplica o cross-encoder se
     RERANKER_ATIVO=true) e gera a resposta com o mesmo
     RAG_PROMPT/chroma/modelo da CP2. Devolve resposta + lista "Fontes:".
  2. classificar_chamado_ti           — LLM com saída Pydantic (reusa os
     enums da CP1 em app/schemas.py) — categoria + prioridade explícita.
  3. gerar_checklist_diagnostico      — LLM com regras de segurança
     (só procedimentos reversíveis).
  4. verificar_informacoes_minimas    — LLM com saída Pydantic; valida se
     o relato contém informações mínimas de triagem (não consulta
     sistemas externos).

Reuso (nada aqui consulta sistemas externos nem cria base nova):
  - Carregar_vectorstore/chroma.db: reaproveitado via buscar_chunks()
    (rag/rag_chain.py) — não reconstrói embeddings por pergunta.
  - Reuso da geração: RAG_PROMPT, build_llm (gemma4:cloud) e
    _formatar_contexto vêm de rag/rag_chain.py.
  - Import primitivo privada (_formatar_contexto) é deliberado: evita
    duplicar o formato de contexto com fonte citada da CP2.
"""

from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from agent.prompts import (
    PROMPT_CHECKLIST,
    PROMPT_CLASSIFICACAO,
    PROMPT_VERIFICACAO,
    REGRAS_PRIORIDADE,
)
from agent.schemas import ClassificacaoChamado, VerificacaoInformacoes
from app.chain import build_llm as build_llm_app  # gemma4:cloud (CP1)
from rag.rag_chain import (
    RAG_PROMPT,
    _formatar_contexto,
    build_llm as build_llm_rag,
    buscar_chunks,
)

# Configuração de chunking escolhida na CP2 (ver ragas_resultado.md).
CONFIG_PADRAO_RAG = "pequeno_512"
K_TRECHOS = 4

INSTRUCAO_INDEXACAO = (
    "A base de conhecimento persistida (chroma_db/) ainda não existe ou "
    "está vazia. Para criar uma única vez com a indexação da CP2:\n"
    "1) abra o Ollama local e confirme o modelo de embeddings "
    "(ollama pull nomic-embed-text);\n"
    "2) na raiz do projeto, rode `python -m rag.main` — na primeira "
    "execução ele lê docs/, gera os e grava em chroma_db/."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def base_indexada(config_nome: str = CONFIG_PADRAO_RAG) -> bool:
    """
    Confere se a coleção do ChromaDB (persistida, mesma da CP2) já tem
    dados, sem disparar geração de embeddings.
    """
    from rag.embeddings_store import carregar_vectorstore

    try:
        vectorstore = carregar_vectorstore(config_nome)
        ids = vectorstore.get(limit=1).get("ids", [])
        return len(ids) > 0
    except Exception:  # noqa: BLE001 — sem base = não indexada
        return False


def _formatar_fontes(trechos: list[Document]) -> str:
    """Lista "Fontes:" com documento, página e classificação de cada trecho
    usado (mesmos metadados preservados pelo ChromaDB da CP2)."""
    linhas = []
    for i, doc in enumerate(trechos, 1):
        fonte = doc.metadata.get("source", "desconhecido")
        pagina = doc.metadata.get("page", "?")
        classificacao = " · ".join(
            str(doc.metadata.get(campo, "—"))
            for campo in ("categoria", "fornecedor", "tipo_documento")
        )
        linhas.append(f"{i}. {fonte} (página {pagina}) — {classificacao}")
    return "\n".join(linhas)


# ---------------------------------------------------------------------------
# Ferramenta 1 — tool RAG (reuso da CP2, busca única)
# ---------------------------------------------------------------------------

@tool
def consultar_base_conhecimento_ti(consulta: str) -> str:
    """
    Busca a resposta para DÚVIDAS TÉCNICAS nos manuais, guias e documentos
    técnicos indexados na base de conhecimento de TI da empresa (Cisco,
    Lenovo, Oracle, CISA — tópicos de rede, hardware, software e
    segurança de acesso). Use quando o usuário perguntar "como fazer",
    "o que é", "quais são as boas práticas" e a resposta deve vir dos
    DOCUMENTOS da empresa, não do conhecimento geral. A resposta vem
    fundamentada nos documentos e, ao final, traz a lista de "Fontes:"
    usadas. NÃO use para cumprimentos, conversa geral ou assuntos
    claramente fora da base de conhecimento.
    """
    if not base_indexada():
        return (
            "A base de conhecimento não está indexada ainda. " +
            INSTRUCAO_INDEXACAO
        )

    try:
        # UMA única recuperação por consulta — o reranking da CP2
        # (cross-encoder) já entra dentro de buscar_chunks().
        trechos = buscar_chunks(consulta, config_nome=CONFIG_PADRAO_RAG, k=K_TRECHOS)
    except Exception as exc:  # noqa: BLE001 — degrada com instrução clara
        return (
            f"[ERRO] Falha ao consultar a base de conhecimento: {exc}\n" +
            INSTRUCAO_INDEXACAO
        )

    if not trechos:
        return (
            "A base de conhecimento não contém informações sobre essa "
            "consulta. Informe ao usuário que o assunto não está coberto "
            "pelos documentos indexados e ofereça abrir chamado — "
            "NÃO responda com conhecimento geral."
        )

    # Geração reusa o MESMO prompt/modelo/chains da CP2 (rag/rag_chain.py).
    llm = build_llm_rag(temperature=0.0)
    resposta = (
        RAG_PROMPT
        | llm
        | StrOutputParser()
    ).invoke({"contexto": _formatar_contexto(trechos), "pergunta": consulta})

    return (
        f"{resposta}\n\n"
        f"Fontes:\n{_formatar_fontes(trechos)}"
    )


# ---------------------------------------------------------------------------
# Ferramenta 2 — classificação de chamado (categoria + prioridade)
# ---------------------------------------------------------------------------

@tool
def classificar_chamado_ti(descricao_problema: str) -> str:
    """
    Classifica o relato de um problema técnico de TI em categoria
    (hardware, software, rede, acesso, outro) e sugere prioridade
    (baixa, media, alta, critica) com base em critérios explícitos de
    impacto e urgência. Use quando o usuário relatar um problema para
    virar chamado ou pedir para classificar a triagem — ex.: "meu notebook
    não conecta no wi-fi desde ontem". Devolve categoria, prioridade,
    justificativa e o que ainda precisa ser perguntado ao usuário. NÃO
    registra o chamado em sistema externo e NÃO inventa dados ausentes.
    """
    parser = PydanticOutputParser(pydantic_object=ClassificacaoChamado)
    prompt = PROMPT_CLASSIFICACAO.partial(
        format_instructions=parser.get_format_instructions(),
        regras_prioridade=REGRAS_PRIORIDADE,
    )
    chain = prompt | build_llm_app(temperature=0.0) | parser

    analise: ClassificacaoChamado = chain.invoke({"descricao": descricao_problema})

    perguntas = (
        "\n".join(f"  - {p}" for p in analise.informacoes_faltantes)
        or "  (nenhuma — relato completo)"
    )
    return (
        f"Categoria: {analise.categoria.value}\n"
        f"Prioridade sugerida: {analise.prioridade.value}\n"
        f"Sistema afetado: {analise.sistema_afetado}\n"
        f"Justificativa: {analise.justificativa}\n"
        f"A perguntar ao usuário:\n{perguntas}\n"
        "Observação: o chamado NÃO foi registrado em nenhum sistema "
        "externo; esta é apenas a classificação da triagem."
    )


# ---------------------------------------------------------------------------
# Ferramenta 3 — checklist de diagnóstico seguro
# ---------------------------------------------------------------------------

@tool
def gerar_checklist_diagnostico(descricao_problema: str) -> str:
    """
    Gera um checklist de verificações iniciais SEGURAS e ordenadas para o
    problema relatado (máximo 6 itens, só procedimentos reversíveis).
    Use quando o usuário pedir o que testar primeiro / passo a passo
    inicial. A entrada pode começar com "Categoria: X" para contextualizar
    (hardware, software, rede, acesso). Proíbe ações destrutivas,
    alterações de permissões e exposição de credenciais. Se faltar
    informação, responde com "PARA ESCLARECER:" em vez de inventar
    procedimentos.
    """
    prompt = PROMPT_CHECKLIST
    chain = prompt | build_llm_app(temperature=0.0) | StrOutputParser()
    return chain.invoke({"entrada": descricao_problema})


# ---------------------------------------------------------------------------
# Ferramenta 4 — verificar informações mínimas do relato
# ---------------------------------------------------------------------------

@tool
def verificar_informacoes_minimas(descricao_problema: str) -> str:
    """
    Verifica se o relato de um problema de TI contém as informações
    mínimas para a triagem (sistema/equipamento afetado, sintoma claro,
    quando começou, impacto/quantas pessoas). Use ANTES de classificar,
    quando o relato ficar confuso/incompleto. Baseia-se EXCLUSIVAMENTE
    no texto fornecido — não consulta sistemas externos. Devolve o que
    está presente e o que falta, cada ausência como pergunta para o
    usuário.
    """
    parser = PydanticOutputParser(pydantic_object=VerificacaoInformacoes)
    prompt = PROMPT_VERIFICACAO.partial(
        format_instructions=parser.get_format_instructions()
    )
    chain = prompt | build_llm_app(temperature=0.0) | parser

    verificacao: VerificacaoInformacoes = chain.invoke(
        {"descricao": descricao_problema}
    )

    presentes = [
        "sistema/equipamento" if verificacao.tem_sistema_ou_equipamento else None,
        "sintoma claro" if verificacao.tem_sintoma_claro else None,
        "início da ocorrência" if verificacao.tem_inicio_da_ocorrencia else None,
        "impacto/criticidade" if verificacao.tem_criticidade_relacionada else None,
    ]
    presentes = [p for p in presentes if p]

    faltando = (
        "\n".join(f"  - {p}" for p in verificacao.informacoes_faltantes)
        or "  (nenhuma)"
    )

    return (
        f"Presentes no relato: {', '.join(presentes) or 'nenhuma'}.\n"
        f"A perguntar ao usuário:\n{faltando}\n"
        f"Resumo: {verificacao.resumo_validade}\n"
        "(Verificação feita apenas sobre o texto fornecido — nada foi "
        "consultado em sistemas externos.)"
    )


# Lista oficial das ferramentas do agente (usada em agent/agent.py).
FERRAMENTAS = [
    consultar_base_conhecimento_ti,
    classificar_chamado_ti,
    gerar_checklist_diagnostico,
    verificar_informacoes_minimas,
]
