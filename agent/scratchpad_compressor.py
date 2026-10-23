"""
scratchpad_compressor.py — Bônus CP3: compressão de contexto do SCRATCHPAD
do agente ReAct (observações acumuladas dentro de UMA execução).

DISTINÇÃO IMPORTANTE — três contextos diferentes, duas técnicas:
  1. Histórico VISUAL (mensagens do gr.Chatbot na tela)  — não é comprimido
     aqui (nem deve); é apenas apresentação.
  2. Histórico de CONVERSA entre turnos (BaseMessage)     — já comprimido
     por aplicar_token_buffer/trim_messages (app/memory_manager.py, CP1).
  3. SCRATCHPAD ReAct (este módulo): dentro de um ÚNICO turno, cada
     chamada de ferramenta acumula um par (Ação, Observação) que o
     AgentExecutor reenvia ao modelo no {agent_scratchpad} a cada iteração.
     Com muitas iterações isso explode o contexto do prompt. Aqui, quando
     o total acumulado passa do limite, as observações ANTIGAS viram um
     RESUMO sintético e apenas as JANELA mais recentes seguem intactas.

PROTOCOLO DE COMPRESSÃO (preservando o que importa):
  - Síntese EXTRATIVA determinística (sem LLM extra): mantém cabeças das
    observações antigas e TODAS as linhas de fonte/fontes VERBATIM — assim
    citações técnicas reais (documento, página, classificação) não são
    perdidas nem re-escritas pelo modelo (sem risco de alucinar citação).
  - As JANELA_RECENTES últimas observações ficam INTACTAS (mesmos objetos).
  - A ação sintética entra com tool="resumo_de_observacoes" e um Thought
    explicando a compactação — o formato ReAct {Thought/Action/Action
    Input/Observation} é preservado, então o parser continua funcionando.

INTEGRAÇÃO COM A VERSÃO ATUAL DO LANGCHAIN (langchain-classic 1.x):
  - O AgentExecutor chama _take_next_step(name_to_tool_map, color_mapping,
    inputs, intermediate_steps, run_manager) a cada iteração (ver fonte
    AgentExecutor._call). A subclasse
    AgentExecutorComCompressao._take_next_step comprime a lista ANTES de
    delegar ao super() — nenhum trecho do executor é reescrito (sem fork).
  - Só o caminho síncrono (.invoke, usado pela CP3) é coberto; os métodos
    assíncronos/Streaming (_aiter_next_step) não são sobrescritos.

Configuração por variável de ambiente (sem credencial):
  - SCRATCHPAD_LIMITE_CARACTERES (padrão 3000): total de caracteres das
    observações a partir do qual a compressão se ativa.
  - SCRATCHPAD_JANELA_RECENTES (padrão 2): quantas observações mais
    recentes permanecem intactas no prompt.
"""

import os
import re
from typing import Any

from langchain_classic.agents import AgentExecutor
from langchain_core.agents import AgentAction

# ---------------------------------------------------------------------------
# Configuração por env (valores default sensatos p/ 4 ferramentas)
# ---------------------------------------------------------------------------

LIMITE_PADRAO_CARACTERES = 3000
JANELA_PADRAO_RECENTES = 2

# Marcador declarativo embutido na observação sintética — usado também
# pelos testes para provar que a compressão de fato aconteceu no prompt.
MARCADOR_COMPRESSAO = (
    "[SÍNTESE COMPACTADA — observações anteriores foram comprimidas para "
    "respeitar o limite de contexto; as '{janela}' observações mais "
    "recentes estão na íntegra abaixo]"
)


def limite_caracteres() -> int:
    try:
        return max(500, int(os.getenv("SCRATCHPAD_LIMITE_CARACTERES", str(LIMITE_PADRAO_CARACTERES))))
    except ValueError:
        return LIMITE_PADRAO_CARACTERES


def janela_recentes() -> int:
    try:
        return max(1, int(os.getenv("SCRATCHPAD_JANELA_RECENTES", str(JANELA_PADRAO_RECENTES))))
    except ValueError:
        return JANELA_PADRAO_RECENTES


# ---------------------------------------------------------------------------
# Extração / síntese determinística (sem LLM — preserva fontes verbatim)
# ---------------------------------------------------------------------------

_PADRAO_FONTE = re.compile(r"(?im)^.*\bf[oa]ntes?\b.*$")


def _extrair_resumo_observacao(
    observacao: str, max_chars: int = 360, max_fontes: int = 600
) -> str:
    """
    Resumo EXTRATIVO de uma observação antiga, em duas partes:
      1. Cabeça: primeiras linhas de conteúdo até max_chars;
      2. Bloco de fontes: da linha "Fonte/Fontes" até o fim, VERBATIM
         (documento + página preservados íntegros, independente do
         orçamento de conteúdo — a citação nunca é perdida nem refeita).
    """
    if not observacao:
        return ""
    linhas = observacao.splitlines()

    indice_fonte = next(
        (i for i, linha in enumerate(linhas) if _PADRAO_FONTE.search(linha)),
        None,
    )
    cabeca = linhas if indice_fonte is None else linhas[:indice_fonte]
    bloco_fontes = [] if indice_fonte is None else linhas[indice_fonte:]

    conteudo = []
    usado = 0
    for linha in cabeca:
        linha_limpa = linha.strip()
        if not linha_limpa:
            continue
        espaco = max_chars - usado
        if espaco <= 0:
            break
        if len(linha_limpa) > espaco:
            # Linha gigante (observação em uma linha só): fatia em vez de
            # descartar — preserva a cabeça técnica.
            conteudo.append(linha_limpa[:espaco].rstrip() + "…")
            usado = max_chars
            break
        conteudo.append(linha_limpa)
        usado += len(linha_limpa)

    partes = [" ".join(conteudo)]
    bloco = "\n".join(linha.strip() for linha in bloco_fontes if linha.strip())
    if bloco:
        partes.append(bloco[:max_fontes])
    return "\n".join(partes)


def _tamanho_passos(passos: list[tuple[Any, str]]) -> int:
    """Total aproximado de caracteres que os passos ocuparão no scratchpad."""
    total = 0
    for acao, observacao in passos:
        total += len(observacao or "")
        total += len(acao.tool or "") + len(acao.tool_input or "")
    return total


def _montar_sintese(passos: list[tuple[AgentAction, str]], janela: int) -> str:
    """Monta a observação sintética dos passos antigos (resumo extrativo)."""
    partes = [MARCADOR_COMPRESSAO.format(janela=janela)]
    for indice, (acao, observacao) in enumerate(passos, 1):
        partes.append(
            f"{indice}. Ferramenta `{acao.tool}` — {_extrair_resumo_observacao(observacao)}"
        )
    sintese = "\n\n".join(partes)
    teto = limite_caracteres()
    # Garantia dura: a própria síntese nunca ultrapassa o limite.
    if len(sintese) > teto:
        sintese = sintese[:teto] + " … [síntese truncada]"
    return sintese


# ---------------------------------------------------------------------------
# Função de compressão + subclasse do executor
# ---------------------------------------------------------------------------

def comprimir_intermediate_steps(
    intermediate_steps: list[tuple[AgentAction, str]],
    limite: int | None = None,
    janela: int | None = None,
) -> list[tuple[AgentAction, str]]:
    """
    Comprime o scratchpad quando ele passa do limite de caracteres:
    passos antigos -> UMA observação sintética; passos recentes intactos.

    Devolve a MESMA lista (mesmos objetos) quando não é preciso comprimir
    (identidade) — só cria objetos novos quando a compressão entra.
    """
    if not intermediate_steps:
        return list(intermediate_steps)

    limite = limite_caracteres() if limite is None else limite
    janela = janela_recentes() if janela is None else janela

    if len(intermediate_steps) <= janela:
        return list(intermediate_steps)  # nada a janelar ainda

    if _tamanho_passos(intermediate_steps) <= limite:
        return list(intermediate_steps)  # dentro do limite: identidade

    antigas = intermediate_steps[:-janela]
    recentes = list(intermediate_steps[-janela:])

    acao_sintetica = AgentAction(
        tool="resumo_de_observacoes",
        tool_input=str(antigas[0][0].tool_input or "")[:120],
        log=(
            "Thought: Passos anteriores foram resumidos para respeitar o "
            "limite de contexto; vou basear-me na síntese e nas observações "
            "recentes na íntegra."
        ),
    )
    observacao_sintetica = _montar_sintese(antigas, janela)

    return [(acao_sintetica, observacao_sintetica)] + recentes


class AgentExecutorComCompressao(AgentExecutor):
    """
    AgentExecutor que comprime AS OBSERVAÇÕES DO SCRATCHPAD antes de cada
    decisão ReAct (veja docstring do módulo para o que é e o que NÃO é).
    Caminho síncrono (.invoke) — o único usado pela CP3.
    """

    def _take_next_step(
        self,
        name_to_tool_map: dict,
        color_mapping: dict,
        inputs: dict,
        intermediate_steps: list[tuple[AgentAction, str]],
        run_manager=None,
    ):
        # Cada iteração reenvia o scratchpad acumulado; comprimimos AQUI,
        # antes de delegar ao AgentExecutor original (sem reescrever o loop).
        comprimidos = comprimir_intermediate_steps(intermediate_steps)
        return super()._take_next_step(
            name_to_tool_map,
            color_mapping,
            inputs,
            comprimidos,
            run_manager,
        )
