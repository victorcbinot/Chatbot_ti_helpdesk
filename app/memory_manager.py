"""
memory_manager.py - 3 estratégias de memória gerenciada (Buffer, Summary,
TokenBuffer) e a estratégia escolhida para o projeto.

NOTA TÉCNICA — por que não usamos langchain.memory clássico:
As classes ensinadas em aula ('ConversationBufferMemory',
'ConversationSummaryMemory', 'ConversationTokenBufferMemory') dependem do
módulo legado 'langchain.chains', que atualmente tem um bug de
compatibilidade conhecido com Python 3.14 (erro ao avaliar type hints do
Pydantic — ver issue oficial:
https://github.com/langchain-ai/langchain/issues/33449). O bug ainda não
tem correção mesmo nas versões mais recentes do LangChain.

Por isso, implementamos aqui as MESMAS 3 estratégias de memória usando
'trim_messages', API nativa e moderna do 'langchain-core' (não depende do
módulo legado). O comportamento e a lógica de cada estratégia são
equivalentes aos ensinados em aula - o que muda é apenas a implementação
por baixo dos panos.
"""

import tiktoken
from langchain_core.messages import BaseMessage, SystemMessage, trim_messages

# Limite de tokens da memória escolhida (dentro da faixa 800–1500 pedida).
TOKEN_BUFFER_LIMIT = 1200

# Usamos tiktoken diretamente como contador de tokens (em vez do tokenizer
# nativo do modelo) porque o contador padrão do langchain-core, quando o
# modelo não expõe um tokenizer próprio (caso do ChatOllama), exige a
# biblioteca `transformers` instalada - dependência pesada e desnecessária
# aqui. Isso também mantém a contagem consistente com a usada em
# context_rot.py.
#
# O encoder é carregado sob demanda (lazy) e não no import do módulo: a
# primeira chamada faz o tiktoken baixar o arquivo do tokenizador da
# internet uma única vez (fica em cache local depois). Carregar de forma
# preguiçosa evita que o projeto inteiro falhe ao importar este módulo
# caso essa rede específica esteja indisponível no momento do import.
_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def _contar_tokens_mensagens(mensagens: list[BaseMessage]) -> int:
    """Conta tokens aproximados de uma lista de mensagens via tiktoken."""
    encoder = _get_encoder()
    return sum(len(encoder.encode(m.content)) for m in mensagens)


def aplicar_buffer(mensagens: list[BaseMessage]) -> list[BaseMessage]:
    """
    Estratégia 1 — Buffer simples.
    Mantém o histórico completo da conversa, sem cortes.
    Prós: nunca perde contexto. Contras: custo de tokens cresce sem limite,
    o que é arriscado em conversas de troubleshooting longas.
    """
    return list(mensagens)


def aplicar_summary(mensagens: list[BaseMessage], llm) -> list[BaseMessage]:
    """
    Estratégia 2 — Summary.
    Resume as mensagens mais antigas usando o próprio LLM e mantém o
    último turno (2 mensagens) na íntegra.
    Prós: custo de tokens baixo e estável. Contras: pode perder detalhes
    técnicos exatos (ex.: um código de erro específico) no resumo.
    """
    if len(mensagens) <= 2:
        return list(mensagens)

    antigas, recentes = mensagens[:-2], mensagens[-2:]
    texto_antigas = "\n".join(f"{m.type}: {m.content}" for m in antigas)

    resumo = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Resuma objetivamente a conversa de suporte técnico "
                    "abaixo, preservando detalhes técnicos importantes "
                    "(sistemas, erros, números de chamado):\n\n"
                    f"{texto_antigas}"
                )
            )
        ]
    )

    resumo_msg = SystemMessage(content=f"Resumo da conversa até aqui: {resumo.content}")
    return [resumo_msg] + list(recentes)


def aplicar_token_buffer(
    mensagens: list[BaseMessage], llm, max_tokens: int = TOKEN_BUFFER_LIMIT
) -> list[BaseMessage]:
    """
    Estratégia 3 — TokenBuffer (ESCOLHIDA para este projeto).
    Mantém as mensagens mais recentes na íntegra até um limite de tokens;
    ao ultrapassar o limite, descarta as mensagens mais antigas.
    Prós: preserva literalmente os detalhes técnicos recentes (essenciais
    em troubleshooting) e ainda assim limita o custo de tokens.
    """
    return trim_messages(
        mensagens,
        max_tokens=max_tokens,
        strategy="last",
        token_counter=_contar_tokens_mensagens,
    )


def aplicar_memoria_do_projeto(mensagens: list[BaseMessage], llm) -> list[BaseMessage]:
    return aplicar_token_buffer(mensagens, llm, max_tokens=TOKEN_BUFFER_LIMIT)