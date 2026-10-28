"""
agent.py — Construção do agente ReAct (CP3).

    Arquitetura:
    ChatOllama (gemma4:cloud, Ollama Cloud via OLLAMA_API_KEY)
        | create_react_agent (prompt ReAct local, langchain_classic)
        | AgentExecutorComCompressao (teto de falhas de formato, limites de
          iteração/tempo; bônus: compressão do scratchpad — ver
          agent/scratchpad_compressor.py)
    Gestão de histórico: lista de BaseMessage + compressão via
    aplicar_token_buffer (app/memory_manager.py da CP1 — trim_messages do
    langchain-core), que corta o histórico antigo acima do limite de
    tokens (TOKEN_BUFFER_LIMIT = 1200) — diferencial de compressão de
    contexto.

IMPORTANTE (ambiente deste projeto): o AgentExecutor e o create_react_agent
NÃO podem vir de "langchain.agents" — no Python 3.14 + Pydantic o módulo
legado quebra na avaliação de type hints (issue langchain-ai/langchain#33449,
documentada em app/memory_manager.py). Aqui eles são importados de
"langchain_classic.agents", validado no venv do projeto."""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_classic.agents import AgentExecutor, create_react_agent

from agent.prompts import REACT_PROMPT
from agent.scratchpad_compressor import AgentExecutorComCompressao, MSG_FALHA_FORMATO
from agent.tools import FERRAMENTAS
from app.chain import build_llm as build_llm_app
from app.memory_manager import aplicar_token_buffer
from agent.guardrail import validar_entrada

# Limites do executor (razoáveis p/ 4 ferramentas e modelo cloud).
MAX_ITERACOES = 8
# O gemma4:cloud (Ollama Cloud) VARIA bastante a latência por chamada
# (documentada em 30–90 s; em janelas lentas passou disso neste projeto).
# Um turno legítimo de 3 chamadas ao LLM já quase esgota 240 s e podia
# parar no limite de tempo mesmo SEM falha — por isso 480 s. O limite de
# iterações continua 8 (o executor também é guardado por
# MAX_FALHAS_FORMATO_CONSECUTIVAS em agent/scratchpad_compressor.py).
MAX_TEMPO_SEGUNDOS = 480

# Saída crua devolvida pelo AgentExecutor (não é exceção!) quando o loop
# se esgota; detectada para virar mensagem amigável na interface.
_SAIDA_STOP_EXECUTOR = "Agent stopped due to iteration limit or time limit"

MSG_FALHA_EXECUCAO = (
    "Não consegui concluir o processamento agora (o modelo pode ter "
    "demorado demais ou respondido de forma inválida). Reenvie a mensagem, "
    "por favor — normalmente funciona no reenvio."
)


def construir_agente() -> AgentExecutor:
    """
    Monta o AgentExecutor ReAct completo: LLM gemma4:cloud + prompt ReAct
    local (agent/prompts.py) + as 4 ferramentas de agent/tools.py.

    Usa AgentExecutorComCompressao (agent/scratchpad_compressor.py): bônus
    de compressão de contexto do scratchpad — observações antigas viram
    síntese quando passam do limite; a janela mais recente fica intacta.
    """
    llm = build_llm_app(temperature=0.0)

    agente_react = create_react_agent(
        llm=llm,
        tools=FERRAMENTAS,
        prompt=REACT_PROMPT,
    )

    return AgentExecutorComCompressao(
        agent=agente_react,
        tools=FERRAMENTAS,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=MAX_ITERACOES,
        max_execution_time=MAX_TEMPO_SEGUNDOS,
        return_intermediate_steps=True,
    )


class AgenteTriagemTI:
    """
    Wrapper de alto nível do agente para a interface: mantém o histórico da
    conversa (comprimido a cada turno) e devolve a resposta final como str.
    """

    def __init__(self) -> None:
        self.executor = construir_agente()
        self.historico: list[BaseMessage] = []

    def _historico_comprimido(self) -> list[BaseMessage]:
        """Aplica a compressão de contexto (TokenBuffer da CP1) ao histórico."""
        return aplicar_token_buffer(self.historico, None)

    def enviar_mensagem(self, mensagem: str) -> str:
        """Envia uma mensagem e devolve só a resposta final (compatibilidade)."""
        resposta, _ferramentas = self.enviar_mensagem_com_passos(mensagem)
        return resposta

    def enviar_mensagem_com_passos(self, mensagem: str) -> tuple[str, list[str]]:
        """
        Envia uma mensagem ao agente e devolve:
          response: resposta final (str);
          ferramentas_usadas: lista RESUMIDA e segura dos nomes (únicos,
              ordem de uso) das ferramentas chamadas — sem o raciocínio
              interno nem o scratchpad do modelo. Usada pela interface web
              para exibir "ferramentas utilizadas" sem expor o Thought.

        GUARDRAIL DE ENTRADA (bônus CP3): validar_entrada() roda ANTES do
        executor — entradas vazias, exageradamente longas, tentativas de
        prompt injection e claramente fora do domínio de TI são recusadas
        educadamente SEM invocar o LLM nem nenhuma ferramenta. À prova de
        erros, a recusa não entra no histórico da conversa.
        """
        veredito = validar_entrada(mensagem)
        if not veredito.aprovado:
            # Registro só no console (sem detalhes sensíveis em tela).
            print(f"[agent] guardrail: entrada recusada ({veredito.motivo})")
            return veredito.mensagem_ao_usuario, []
        ferramentas_usadas: list[str] = []
        ferramentas_validas = {t.name for t in FERRAMENTAS}
        try:
            resultado = self.executor.invoke(
                {
                    "input": mensagem,
                    "chat_history": self._historico_comprimido(),
                }
            )
            resposta = resultado.get("output", "").strip()
            # intermediate_steps = [((tool_call), observação), ...]; capturamos
            # APENAS o nome da ferramenta (registro resumido e seguro).
            vistos: set[str] = set()
            for chamada, _observacao in resultado.get(
                "intermediate_steps", []
            ):
                nome = chamada.tool
                # Ignora passos internos do executor (ex.: "_Exception"
                # quando o handle_parsing_errors corrige o formato ReAct):
                # só contam ferramentas REAIS registradas em FERRAMENTAS.
                if nome and nome in ferramentas_validas and nome not in vistos:
                    vistos.add(nome)
                    ferramentas_usadas.append(nome)

            # O AgentExecutor NÃO lança exceção ao esgotar iterações/tempo:
            # devolve a frase crua como "output" — assim como a resposta
            # vazia (modelo cloud sem saída útil) e o encerramento pelo
            # teto de falhas de formato do scratchpad. Em todos esses
            # casos: mensagem amigável e o turno NÃO entra no histórico
            # (estado consistente para a próxima mensagem).
            if (
                not resposta
                or resposta.startswith(_SAIDA_STOP_EXECUTOR)
                or resposta.startswith(MSG_FALHA_FORMATO)
            ):
                print(f"[agent] turno sem resposta usável do modelo "
                      f"(ferramentas: {ferramentas_usadas or 'nenhuma'})")
                return MSG_FALHA_EXECUCAO, ferramentas_usadas
        except Exception as exc:  # noqa: BLE001 — msg amigável s/ detalhes sensíveis
            print(f"[agent] erro de execução: {exc!r}")  # detalhe só no console
            # O turno falhou: NÃO entra no histórico (evita poluir o
            # contexto da próxima mensagem com um desabafo de sistema).
            return MSG_FALHA_EXECUCAO, []

        self.historico.append(HumanMessage(content=mensagem))
        self.historico.append(AIMessage(content=resposta))
        return resposta, ferramentas_usadas

    def limpar_conversa(self) -> None:
        """Limpa o histórico da sessão (usado pela interface da CP3)."""
        self.historico = []
