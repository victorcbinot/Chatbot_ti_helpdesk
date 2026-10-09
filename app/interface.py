"""
interface.py — Interface web (Streamlit) do DocMind RAG.

Reutiliza o pipeline existente em rag/ (buscar_com_fontes em
rag/rag_chain.py) para responder e exibir as fontes — sem duplicar
carregamento, embeddings, recuperação ou geração.

Execução (a partir da raiz do projeto):
    streamlit run app/interface.py
"""

import sys
from pathlib import Path

# O Streamlit executa o script de app/, um nível abaixo da raiz onde está o
# pacote rag/. Garante a raiz no sys.path para os imports funcionarem.
RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import streamlit as st  # noqa: E402

from rag.main import CONFIG_PADRAO  # noqa: E402
from rag.metadata import (  # noqa: E402
    construir_filtro,
    descrever_filtro,
    valores_disponiveis,
)
from rag.rag_chain import buscar_com_fontes  # noqa: E402
from rag.reranker import reranker_ativo  # noqa: E402
from rag.splitter import CONFIGURACOES_CHUNKING  # noqa: E402

st.set_page_config(page_title="DocMind RAG — Assistente de TI", page_icon="🛠️")

st.title("DocMind RAG — Assistente de TI")
st.caption(
    "Respondo com base nos documentos técnicos de docs/ "
    "(nomic-embed-text local + ChromaDB + gemma4:cloud) e cito a fonte."
)

# ------------------------- Sidebar: configurações -------------------------

st.sidebar.header("Configurações do pipeline")

config_nome = st.sidebar.selectbox(
    "Configuração de chunking",
    options=list(CONFIGURACOES_CHUNKING.keys()),
    index=list(CONFIGURACOES_CHUNKING.keys()).index(CONFIG_PADRAO),
    help="Uma coleção por configuração no ChromaDB (pequeno_512 ou grande_1024).",
)

usar_rerank = st.sidebar.toggle(
    "Reranking (cross-encoder)",
    value=reranker_ativo(),
    help="Reordena os candidatos recuperados por relevância antes da geração.",
)

st.sidebar.markdown("**Filtro de metadata (opcional)**")

selecionar = {}
for campo in ("categoria", "fornecedor", "tipo_documento"):
    opcoes = ["Sem filtro"] + valores_disponiveis(campo)
    selecionar[campo] = st.sidebar.selectbox(
        campo.replace("_", " ").title(),
        options=opcoes,
        help="Restringe a busca aos metadados registrados dos documentos.",
    )

filtro = construir_filtro(
    categoria=None if selecionar["categoria"] == "Sem filtro" else selecionar["categoria"],
    fornecedor=None if selecionar["fornecedor"] == "Sem filtro" else selecionar["fornecedor"],
    tipo_documento=None if selecionar["tipo_documento"] == "Sem filtro" else selecionar["tipo_documento"],
)
if filtro:
    st.sidebar.caption(f"Filtro ativo: {descrever_filtro(filtro)}")
else:
    st.sidebar.caption("Filtro ativo: sem filtro")

if st.sidebar.button("Limpar conversa", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# ------------------------- Histórico da conversa -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Olá! Sou o DocMind. Pergunte sobre diagnóstico de rede, "
                "manutenção de hardware, troubleshooting de aplicações Java ou "
                "segurança de acesso — vou responder com base nos documentos e "
                "mostrar as fontes. (Comandos como `fontes`, `config` e `filtro` "
                "são do chat de terminal; aqui ajuste tudo na barra lateral.)"
            ),
            "fontes": [],
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("fontes"):
            with st.expander(f"Fontes usadas ({len(msg['fontes'])})"):
                for i, doc in enumerate(msg["fontes"], 1):
                    meta = doc.metadata
                    st.markdown(
                        f"**{i}.** `{meta.get('source', 'desconhecido')}` "
                        f"— página {meta.get('page', '?')}"
                    )
                    st.caption(
                        f"{meta.get('categoria', '—')} · "
                        f"{meta.get('fornecedor', '—')} · "
                        f"{meta.get('tipo_documento', '—')}"
                    )
                    st.markdown(f"{doc.page_content[:300]}…")

# --------------------------- Entrada e resposta ---------------------------

def _mensagem_erro() -> str:
    return (
        "Não foi possível gerar a resposta agora. Verifique: 1) o Ollama local "
        "está rodando com o modelo nomic-embed-text; 2) a variável "
        "OLLAMA_API_KEY está configurada no .env; 3) há conexão com a Ollama "
        "Cloud. Se o problema persistir, confira os logs no terminal."
    )


if pergunta := st.chat_input("Pergunte sobre rede, hardware, software ou segurança de acesso…"):
    st.session_state.messages.append({"role": "user", "content": pergunta})

    with st.chat_message("user"):
        st.markdown(pergunta)

    resposta_final = ""
    fontes_final: list = []

    with st.chat_message("assistant"):
        with st.spinner("Consultando a base de conhecimento e gerando a resposta…"):
            try:
                resposta, trechos = buscar_com_fontes(
                    pergunta,
                    config_nome=config_nome,
                    filtro=filtro,
                    rerank=usar_rerank,
                )
                if not trechos:
                    resposta = (
                        "Não encontrei informações suficientes nos documentos da "
                        "base para responder a essa pergunta. Tente reformular "
                        "ou perguntar sobre rede, hardware, software ou acesso."
                    )
                st.markdown(resposta)
                resposta_final = resposta
                fontes_final = trechos

                if trechos:
                    with st.expander(f"Fontes usadas ({len(trechos)})"):
                        for i, doc in enumerate(trechos, 1):
                            meta = doc.metadata
                            st.markdown(
                                f"**{i}.** `{meta.get('source', 'desconhecido')}` "
                                f"— página {meta.get('page', '?')}"
                            )
                            st.caption(
                                f"{meta.get('categoria', '—')} · "
                                f"{meta.get('fornecedor', '—')} · "
                                f"{meta.get('tipo_documento', '—')}"
                            )
                            st.markdown(f"{doc.page_content[:300]}…")

            except Exception as exc:  # noqa: BLE001 — mensagem amigável ao usuário
                texto_erro = _mensagem_erro()
                st.error(texto_erro)
                resposta_final = texto_erro
                # Detalhe vai para o console (sem expor credenciais/chaves).
                print(f"[interface] erro ao responder: {exc!r}")

    st.session_state.messages.append(
        {"role": "assistant", "content": resposta_final, "fontes": fontes_final}
    )