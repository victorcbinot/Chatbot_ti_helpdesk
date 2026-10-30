"""
interface_gradio.py — Interface web da CP3 (Gradio).

Execução (raiz do projeto):
    python -m agent.interface_gradio

Camada de APRESENTAÇÃO apenas: a conversa roda em AgenteTriagemTI
(agent/agent.py), que já encapsula o AgentExecutor ReAct, as 4 ferramentas
(agent/tools.py) e o pipeline RAG da CP2 (rag/) — nada de RAG, recuperação
ou lógica de agente é replicada aqui.

Comportamentos garantidos:
  - Histórico POR SESSÃO: cada navegador que abrir a página recebe sua
    própria instância de AgenteTriagemTI em um gr.State (Gradio cria um
    estado novo por sessão) — não há estado compartilhado entre usuários
    e não há duplicação de histórico (a tela mostra mensagens de UI e o
    agente mantém o contexto comprimido, cada um com seu papel).
  - Botão "Limpar conversa": zera o chat da tela E o histórico interno
    do agente da sessão (AgenteTriagemTI.limpar_conversa).
  - Fontes: quando a resposta contém a seção "Fontes:" (gerada pela tool
    RAG da CP2 com os metadados reais do ChromaDB), ela é exibida em um
    painel próprio.
  - Ferramentas usadas: exibição APENAS de registro resumido (nomes
    únicos) — o raciocínio interno (Thought/scratchpad) nunca é exibido.
  - Erros: exceções são capturadas e viram mensagens amigáveis na tela;
    detalhes técnicos vão somente para o console (sem chaves, stack
    traces ou informações sensíveis).

A abertura cria a URL local e uma URL pública temporária (share=True).
A URL pública vale SÓ enquanto o processo estiver em execução (ver README).
"""

import re
import sys
from pathlib import Path

# Como em app/interface.py (Streamlit), o Gradio executa o script de dentro
# do pacote; garante a raiz do projeto no sys.path para importar agent/,
# rag/ e app/ sem depender do diretório de chamada.
RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import gradio as gr  # noqa: E402

from agent.agent import AgenteTriagemTI  # noqa: E402
from agent.tools import INSTRUCAO_INDEXACAO, base_indexada  # noqa: E402

# ---------------------------------------------------------------------------
# Erro de configuração de sessão (sem detalhes sensíveis na tela)
# ---------------------------------------------------------------------------

ERRO_SESSAO = (
    "Não foi possível iniciar a sessão do agente. Causas prováveis:\n\n"
    "1. **Chave da Ollama Cloud ausente** — verifique o arquivo `.env` na "
    "raiz do projeto (a variável de credencial está documentada em "
    "`.env.example`);\n"
    "2. **Ollama local não está rodando** — abra o Ollama e confirme o "
    "modelo de embeddings (`ollama pull nomic-embed-text`).\n\n"
    "Os detalhes técnicos estão no terminal onde este app foi iniciado."
)

# Ao subir a interface, só AVISA se a base não estiver indexada — não
# bloqueia o chat: a tool RAG devolve as instruções de indexação se a
# consulta precisar da base.
if not base_indexada():
    print(f"[interface] {INSTRUCAO_INDEXACAO}")


# ---------------------------------------------------------------------------
# Funções de apresentação (chamam o agente; só organizam a UI)
# ---------------------------------------------------------------------------

def _nova_sessao_agente() -> AgenteTriagemTI | None:
    """
    Cria a instância de agente da sessão (lazy — depende da chave no .env).
    Em caso de falha de configuração, devolve None (o próximo turno tenta
    de novo) e registra o detalhe apenas no console.
    """
    try:
        return AgenteTriagemTI()
    except Exception as exc:  # noqa: BLE001
        print(f"[interface] falha ao iniciar agente da sessão: {exc!r}")
        return None


_PADRAO_SECAO_FONTES = re.compile(
    r"(?im)^[#* \-]*(\*\*)?[fF]ontes?(\*\*)?:?[#* \-]*$"
)
# Formato alternativo: o LLM pode resumir a seção da tool em uma citação
# INLINE no meio do parágrafo (ex.: "... Fonte: 01_x.pdf, página 10."), com
# conteúdo na MESMA linha do rótulo — a seção dedicada não casa nesse caso.
_PADRAO_FONTES_INLINE = re.compile(
    r"(?im)[#*> \-]*(\*\*)?[fF]ontes?(\*\*)?:\s*(\*\*)?[ \t]*([^\n]+?)[ \t]*\**[ \t]*$"
)


def _extrair_fontes(resposta: str) -> str:
    """
    Recorta, da resposta final do agente, o bloco de fontes montado pela
    tool RAG da CP2 ("Fontes:" + documento · página · classificação reais
    do ChromaDB). Aceita os dois formatos que a resposta pode assumir:
      1. seção dedicada (linha contendo apenas "Fontes:" e o bloco abaixo);
      2. citação INLINE ("Fonte: doc.pdf, página N.") no meio do texto.
    Sem nenhuma ocorrência, devolve string vazia.
    """
    secao = _PADRAO_SECAO_FONTES.search(resposta)
    if secao is not None:
        return resposta[secao.end():].strip()
    inline = _PADRAO_FONTES_INLINE.search(resposta)
    if inline is None:
        return ""
    return inline.group(4).strip()


def _formatar_ferramentas(ferramentas: list[str]) -> str:
    """Registro resumido e SEGURO das ferramentas usadas (apenas nomes)."""
    if not ferramentas:
        return "Nenhuma — o agente respondeu diretamente."
    lista = "\n".join(f"- `{nome}`" for nome in ferramentas)
    return f"O agente utilizou {len(ferramentas)} ferramenta(s):\n\n{lista}"


def responder(
    mensagem: str,
    historico_tela: list | None,
    agente_da_sessao: AgenteTriagemTI | None,
):
    """
    Um turno da conversa.

    Entrada:  (Textbox, Chatbot, gr.State com o agente da sessão)
    Saída:    (Chatbot, gr.State, Textbox, Fontes, Ferramentas)

    O histórico da TELA (gr.ChatMessage) é independente do histórico do
    AGENTE (BaseMessage comprimido dentro de AgenteTriagemTI); cada sessão
    tem o seu, sem compartilhamento entre usuários.
    """
    historico_tela = list(historico_tela or [])
    historico_tela.append(gr.ChatMessage(role="user", content=mensagem))

    if agente_da_sessao is None:
        agente_da_sessao = _nova_sessao_agente()
    if agente_da_sessao is None:
        historico_tela.append(gr.ChatMessage(role="assistant", content=ERRO_SESSAO))
        return (
            historico_tela,
            agente_da_sessao,
            "",
            ERRO_SESSAO,
            "Nenhuma ferramenta pôde ser executada.",
        )

    resposta, ferramentas = agente_da_sessao.enviar_mensagem_com_passos(mensagem)
    historico_tela.append(gr.ChatMessage(role="assistant", content=resposta))

    bloco_fontes = _extrair_fontes(resposta)
    painel_fontes = (
        bloco_fontes
        if bloco_fontes
        else "Nenhuma fonte consultada — esta resposta não usou a base de conhecimento."
    )

    return (
        historico_tela,
        agente_da_sessao,
        "",
        painel_fontes,
        _formatar_ferramentas(ferramentas),
    )


def limpar_conversa(agente_da_sessao: AgenteTriagemTI | None):
    """
    Botão "Limpar conversa": zera a tela E o histórico INTERNO do agente
    desta sessão (não troca de instância nem afeta outros usuários).
    """
    if agente_da_sessao is not None:
        agente_da_sessao.limpar_conversa()
    return (
        [],
        agente_da_sessao,
        "",
        "Nenhuma fonte consultada.",
        "Nenhuma ferramenta ainda.",
    )


# ---------------------------------------------------------------------------
# Montagem da UI (gr.Blocks)
# ---------------------------------------------------------------------------

DESCRICAO = (
    "Converse em português: relate um problema de TI (computador, rede, "
    "sistema, acesso) ou faça perguntas técnicas. O agente responde com "
    "base nos manuais e guias internos da empresa e cita as fontes. "
    "Sua conversa é individual: este histórico só existe na sua sessão."
)

EXEMPLOS = [
    "O que é autenticação multifator resistente a phishing?",
    "Meu notebook não liga e preciso dele hoje para uma reunião importante.",
    "Quem é você e o que consegue fazer?",
]


def construir_interface() -> gr.Blocks:
    """Monta a UI completa (chat + fontes + ferramentas + limpar)."""
    with gr.Blocks(title="DocAgent de TI — CP3") as interface:
        gr.Markdown(f"# DocAgent de TI (CP3)\n\n{DESCRICAO}")

        estado_agente = gr.State(None)

        chat_tela = gr.Chatbot(
            label="Conversa com o assistente de TI",
            height=430,
        )

        with gr.Accordion("Fontes consultadas nesta resposta", open=False):
            fonte_mark = gr.Markdown("Nenhuma fonte consultada.")

        with gr.Accordion("Ferramentas usadas nesta resposta", open=False):
            ferramentas_mark = gr.Markdown("Nenhuma ferramenta ainda.")

        with gr.Row():
            caixa = gr.Textbox(
                placeholder="Digite sua mensagem em português e pressione Enter…",
                label="Sua mensagem",
                scale=9,
                max_lines=4,
            )
            botao_enviar = gr.Button("Enviar", variant="primary", scale=2)

        botao_limpar = gr.Button("Limpar conversa", variant="secondary")

        gr.Examples(examples=EXEMPLOS, inputs=[caixa])

        saidas_turno = [chat_tela, estado_agente, caixa, fonte_mark, ferramentas_mark]

        caixa.submit(
            responder, [caixa, chat_tela, estado_agente], saidas_turno
        )
        botao_enviar.click(
            responder, [caixa, chat_tela, estado_agente], saidas_turno
        )
        botao_limpar.click(
            limpar_conversa,
            [estado_agente],
            [chat_tela, estado_agente, caixa, fonte_mark, ferramentas_mark],
        )

    return interface


if __name__ == "__main__":
    # share=True cria a URL pública temporária do Gradio. Ela depende da
    # internet e vale apenas enquanto este processo estiver em execução —
    # ao encerrar o app, a URL deixa de funcionar (ver README, seção CKP03).
    construir_interface().launch(share=True)
