"""
Experimento:
1. Plantamos um dado crítico logo no início da conversa (o número de um
   chamado de TI: "TI-48213").
2. Preenchemos a conversa com turnos de troubleshooting genéricos e
   irrelevantes (o "ruído"), em quantidades crescentes: 0, 5, 10, 15, 20.
3. Ao final de cada versão, perguntamos: "Qual foi o número do chamado que
   eu informei no início da nossa conversa?"
4. Registramos: quantidade de tokens da janela, se o modelo acertou o
   número, e o tempo/latência da chamada.
5. Geramos uma tabela comparativa e, opcionalmente, um gráfico (requisito
   diferencial "Context engineering com métricas").
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import tiktoken
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.chain import build_llm
from app.prompts import SYSTEM_PROMPT_CHAT

CHAMADO_PLANTADO = "TI-48213"
PERGUNTA_FINAL = (
    "Qual foi o número do chamado que eu informei logo no início da "
    "nossa conversa? Responda apenas com o número."
)

# Quantidade de turnos de "ruído" (troubleshooting genérico) injetados
# antes da pergunta final, para simular contexto crescente.
NIVEIS_DE_RUIDO = [0, 5, 10, 15, 20]

_ENCODER = tiktoken.get_encoding("cl100k_base")


def _contar_tokens(mensagens: list) -> int:
    total = 0
    for m in mensagens:
        total += len(_ENCODER.encode(m.content))
    return total


def _gerar_turno_ruido(indice: int) -> tuple[HumanMessage, AIMessage]:
    pergunta = (
        f"Também estou com lentidão ao abrir o aplicativo interno número "
        f"{indice}, isso é normal?"
    )
    resposta = (
        f"Lentidão no aplicativo {indice} pode ter várias causas. Tente "
        f"limpar o cache e verificar sua conexão de rede."
    )
    return HumanMessage(content=pergunta), AIMessage(content=resposta)


def _montar_conversa(qtd_ruido: int) -> list:
    mensagens = [
        SystemMessage(content=SYSTEM_PROMPT_CHAT),
        HumanMessage(
            content=(
                f"Estou abrindo um chamado, o número dele é {CHAMADO_PLANTADO}. "
                "Meu notebook não liga."
            )
        ),
        AIMessage(
            content=(
                f"Entendido, registrei o chamado {CHAMADO_PLANTADO}. Pode me "
                "dizer se algum LED acende ao apertar o botão de energia?"
            )
        ),
    ]

    for i in range(qtd_ruido):
        h, a = _gerar_turno_ruido(i + 1)
        mensagens.append(h)
        mensagens.append(a)

    mensagens.append(HumanMessage(content=PERGUNTA_FINAL))
    return mensagens


@dataclass
class ResultadoContextRot:
    turnos_de_ruido: int
    tokens_na_janela: int
    resposta_modelo: str
    acertou: bool
    latencia_segundos: float


def rodar_experimento_context_rot() -> list[ResultadoContextRot]:
    llm = build_llm(temperature=0.0)
    resultados: list[ResultadoContextRot] = []

    for qtd_ruido in NIVEIS_DE_RUIDO:
        mensagens = _montar_conversa(qtd_ruido)
        tokens = _contar_tokens(mensagens)

        inicio = time.perf_counter()
        resposta = llm.invoke(mensagens)
        latencia = time.perf_counter() - inicio

        texto_resposta = resposta.content.strip()
        acertou = CHAMADO_PLANTADO in texto_resposta

        resultados.append(
            ResultadoContextRot(
                turnos_de_ruido=qtd_ruido,
                tokens_na_janela=tokens,
                resposta_modelo=texto_resposta,
                acertou=acertou,
                latencia_segundos=round(latencia, 2),
            )
        )

    return resultados


def imprimir_tabela(resultados: list[ResultadoContextRot]) -> None:
    cabecalho = (
        f"{'Turnos ruído':>12} | {'Tokens':>7} | {'Acertou?':>8} | "
        f"{'Latência (s)':>12} | Resposta do modelo"
    )
    print(cabecalho)
    print("-" * len(cabecalho))
    for r in resultados:
        status = "SIM" if r.acertou else "NÃO"
        print(
            f"{r.turnos_de_ruido:>12} | {r.tokens_na_janela:>7} | "
            f"{status:>8} | {r.latencia_segundos:>12} | {r.resposta_modelo[:60]}"
        )


def gerar_grafico(resultados: list[ResultadoContextRot], caminho_saida: str = "context_rot.png") -> None:

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tokens = [r.tokens_na_janela for r in resultados]
    cores = ["#2ecc71" if r.acertou else "#e74c3c" for r in resultados]
    rotulos = [f"{r.turnos_de_ruido} turnos" for r in resultados]

    plt.figure(figsize=(8, 5))
    plt.bar(rotulos, tokens, color=cores)
    plt.ylabel("Tokens na janela de contexto")
    plt.title("Context Rot — verde: acertou o dado plantado | vermelho: errou")
    plt.tight_layout()
    plt.savefig(caminho_saida)
    print(f"Gráfico salvo em: {caminho_saida}")


if __name__ == "__main__":
    resultados = rodar_experimento_context_rot()
    imprimir_tabela(resultados)
    try:
        gerar_grafico(resultados)
    except ImportError:
        print("matplotlib não instalado — pulando geração do gráfico.")