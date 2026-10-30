# CKP01 - Chatbot Profissional · Assistente de Triagem de Chamados de TI

**Prompt Engineering & AI · FIAP · 2º Semestre 2026**

> **Relação com as demais CPs:** esta é a documentação da CP1 (branch `cp1`). A CP2 (RAG) está documentada na branch `cp2` e a CP3 (agente) na branch `cp3`; Aqui vive só o que é da CP1 — o código desta etapa está em `app/` e não foi alterado pelas demais checkpoints.

**Integrantes:** 
| Nome | RM |
|-|-|
| Gustavo Kunitaki | 571400 |
| Pedro Ferreras | 568713 |
| Pedro Santos | 571017 |
| Victor Binot | 571499 |


## Domínio

**Domínio escolhido:** Triagem de chamados de suporte técnico de TI (helpdesk corporativo).

**Por que foi escolhido:** é um domínio com vocabulário técnico bem definido
(o que facilita manter a persona e evitar que o chatbot "saia do
personagem"), gera naturalmente saídas estruturáveis (categoria, urgência,
sistema afetado) e se conecta de forma direta com o restante do semestre:
no CKP02 esse chatbot vira a base de um RAG sobre uma base de conhecimento
de TI (KB de erros comuns, manuais, políticas de acesso), e no CKP03 vira
uma tool de um agente capaz de abrir chamados e consultar status de
tickets.

**Usuários-alvo:** colaboradores internos de uma empresa que precisam
relatar problemas técnicos (hardware, software, rede ou acesso) antes de
serem atendidos por um técnico humano.

## Requisitos atendidos

| Requisito | Status | Implementação |
|-|-|-|
| Pipeline LCEL | ✅ | `chain.py` — `prompt \| llm \| PydanticOutputParser()` (chain de extração) |
| ChatOllama | ✅ | `gemma4:cloud` via Ollama Cloud, chave em `.env` |
| Arquitetura de 2 chains (Aula 03) | ✅ | `ConversationTI` (chat com memória, LCEL) + pipeline LCEL (extração) - ver nota técnica abaixo |
| Memória gerenciada | ✅ | TokenBuffer via `trim_messages` + `tiktoken`, justificada abaixo |
| Pydantic v2 (≥4 campos) | ✅ | `AnaliseChamado` com 6 campos em `schemas.py` |
| Context rot | ✅ | `context_rot.py` - tabela com 0/5/10/15/20 turnos de ruído |
| System prompt com persona | ✅ | `prompts.py` - XML tagging (`<persona>`, `<regras>`, `<restricoes>`) |
| Domínio documentado | ✅ | Este README |
| Interface | ✅ | CLI (terminal) via `app/main.py` - sem dependência de navegador/servidor web |
| Context engineering com métricas (diferencial) | ✅ | `context_rot.py` - contagem de tokens (tiktoken) + gráfico (matplotlib) |
| Meta prompting (diferencial) | ✅ | `meta_prompting.py` - rode e cole o resultado na seção abaixo |

## Como executar (local - sem Colab)

```bash
cp .env.example .env      # edite com sua OLLAMA_API_KEY 
pip install -r requirements.txt
python -m app.main        # abre o chat direto no terminal
```

Durante a conversa, digite `analisar` a qualquer momento para gerar a análise
estruturada do chamado, ou `sair` para encerrar.

Para rodar o experimento de context rot separadamente:

```bash
python -m app.context_rot
```

## Funcionalidades e detalhes técnicos

- **Chat de conversação** (`ConversationTI`, em `app/chain.py`): system prompt
  (`app/prompts.py`) + memória de conversa + `gemma4:cloud` em uma chain
  LCEL (`prompt | llm`). O chat usa `temperature=0.3`; a extração estruturada
  usa `temperature=0.0`.
- **Comando `analisar`** (`app/main.py`): roda a **Chain 2 de extração
  estruturada** — pipeline LCEL `prompt | llm | PydanticOutputParser`,
  com prompts dedicados (`SYSTEM_PROMPT_EXTRACAO`).
- **Guardrail de saída com Pydantic** (`app/schemas.py`): o modelo
  `AnaliseChamado` obrigatoriamente valida a saída em campos tipados —
  `categoria` (enum: hardware/software/rede/acesso/outro), `urgencia`
  (enum: baixa/média/alta/crítica), `sistema_afetado`, `resumo_problema`,
  `acao_recomendada` e `requer_escalonamento` — com `field_validator`
  que recusa campos de texto vazios. Saída fora do formato falha no parser
  e não chega ao usuário.
- **Prompt com restrições** (`app/prompts.py`): persona de assistente de TI,
  máximo de 2 passos de troubleshooting básico antes de escalar, proibição
  de armazenar senhas, recusa de assuntos fora de TI e manutenção da persona
  sob jailbreak. Sem ferramenta e sem RAG nesta etapa: as respostas vêm
  apenas do modelo guiado pelo prompt de sistema.

## Nota técnica - por que não usamos `ConversationChain` literal

As classes clássicas ensinadas em aula (`ConversationChain`,
`ConversationBufferMemory`, `ConversationSummaryMemory`,
`ConversationTokenBufferMemory`) dependem do módulo legado
`langchain.chains`/`langchain.memory`, que tem um **bug de compatibilidade
conhecido e ainda sem correção** com Python 3.14 (falha ao avaliar type
hints do Pydantic - ver issue oficial do LangChain:
https://github.com/langchain-ai/langchain/issues/33449).

Para não depender de downgrade de Python, reimplementamos a **mesma
arquitetura de 2 chains** com APIs modernas e nativas do `langchain-core`:

- **Chain 1 (chat):** classe `ConversationTI` (`app/chain.py`), que
  encapsula uma chain LCEL (`prompt | llm`, com o operador `|`) e aplica a
  memória gerenciada (TokenBuffer, via `trim_messages`) sobre o histórico
  a cada turno - mesmo papel do `ConversationChain` das aulas.
- **Chain 2 (extração):** pipeline LCEL 
  `prompt | llm | PydanticOutputParser()`, sem alterações.

O comportamento funcional é equivalente ao ensinado; apenas a
implementação por baixo dos panos é mais moderna e compatível com
qualquer versão do Python.

## Justificativa da memória

**Estratégia escolhida:** TokenBuffer (`ConversationTokenBufferMemory`),
com limite de **1200 tokens**.

**Por quê:** em uma conversa de troubleshooting de TI, os detalhes técnicos
mais recentes (último erro relatado, último passo testado pelo usuário)
são os mais importantes para a próxima resposta do assistente - muito mais
do que o início da conversa. O TokenBuffer preserva essas mensagens recentes
**literalmente** (sem risco de um resumo automático "perder" um código de
erro ou nome de sistema específico), e ainda assim limita o custo de tokens
ao descartar mensagens antigas quando o limite é ultrapassado. Comparado ao
Summary, evitamos a perda de detalhes técnicos exatos; comparado ao Buffer,
evitamos o crescimento descontrolado do custo em conversas longas.

**Contador de tokens:** `tiktoken` em modo lazy (baixa o vocabulário só no
primeiro uso), em vez do contador do LangChain — que, quando o modelo não
expõe tokenizer próprio (caso do `ChatOllama`), exigiria a dependência
pesada `transformers`. O mesmo contador é usado em `context_rot.py`, o que
mantém a contagem consistente entre chat e experimento.

## Estrutura do projeto

```
Chatbot_ti_helpdesk/
├── app/
│   ├── __init__.py
│   ├── main.py            
│   ├── chain.py            
│   ├── memory_manager.py
|   ├── meta_prompting.py  
│   ├── schemas.py          
│   ├── context_rot.py      
│   └── prompts.py          
├── .env.example
├── requirements.txt
├── meta_prompting_resultado.md
└── README.md
```


## Context engineering com métricas (diferencial) — experimento `context_rot.py`

1. Planta um dado crítico no início da conversa: o número do chamado
   **TI-48213**.
2. Preenche a conversa com turnos genéricos de troubleshooting ("ruído"),
   em níveis crescentes: **0, 5, 10, 15, 20** turnos.
3. Ao final de cada versão, pergunta: "Qual foi o número do chamado que eu
   informei no início da nossa conversa?"
4. Registra: quantidade de tokens da janela (tiktoken), se o modelo acertou
   o número e a latência da chamada.
5. Imprime a tabela comparativa e gera o gráfico `context_rot.png`
   (matplotlib; sem `matplotlib` instalado, o script pula só o gráfico).

O resultado liga o experimento à escolha da memória TokenBuffer: limitar a
janela por tokens preserva os dados críticos quando o contexto longo começa
a "apodrecer" (context rot).

## Meta prompting (diferencial)

Rode:

```bash
python -m app.meta_prompting
```

Isso usa o próprio modelo (gemma4:cloud) para criticar o system prompt
atual (`SYSTEM_PROMPT_CHAT`) e sugerir uma versão melhorada, salvando tudo
em `meta_prompting_resultado.md`.

O resultado já produzido fica registrado em `meta_prompting_resultado.md`
(projeto versionado), com o **antes/depois** do prompt lado a lado: o
"DEPOIS" revisado traz triagem inicial com coleta obrigatória de
SO/equipamento/tempo, limitação rigorosa do troubleshooting a 2 tentativas
antes de escalar, formato fixo de "Resumo para Técnica" e restrições
reforçadas de segurança e persona. O prompt efetivamente usado em
produção (`app/prompts.py`) é o original com a seção `<restricoes>`
(adicionada na revisão), e o `SYSTEM_PROMPT_EXTRACAO` é um prompt dedicado
à extração estruturada.
