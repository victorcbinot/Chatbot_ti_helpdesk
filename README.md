# Chatbot de TI — DocMind · CP1 + CP2 + CP3

**Prompt Engineering & AI · FIAP · 2º Semestre 2026**

**Projeto:** assistente de triagem de chamados de suporte técnico de TI. A base é a CP1 (chatbot com guardrails de saída via Pydantic) e a CP2 (RAG com ChromaDB, metadata filtering, reranking e avaliação RAGAS). A **CP3** (seção [CKP03](#ckp03--agente-de-triagem-de-ti-react--agentexecutor)) transforma o RAG em *tool* de um agente ReAct com `AgentExecutor`, interface Gradio e três bônus (input guardrail, compressão de contexto em dois níveis e mais de 3 ferramentas).

**Integrantes:**

| Nome | RM |
|---|---|
| Gustavo Kunitaki | 571400 |
| Pedro Ferreras | 568713 |
| Pedro Santos | 571017 |
| Victor Binot | 571499 |

## Domínio

**Domínio:** triagem de chamados de suporte técnico de TI (helpdesk corporativo) — o mesmo do CKP01.

**Por que foi escolhido:** é um domínio com documentação técnica real, pública e confiável (fabricantes, órgãos de segurança), o que permite montar uma base de conhecimento verdadeira. No CKP01 o chatbot coletava o relato do problema; aqui ele passa a **responder com base em documentos técnicos reais**, citando a fonte. No CKP03 o pipeline vira uma *tool* (`buscar(consulta)`) de um agente.

**Usuários-alvo:** colaboradores e analistas de suporte de uma empresa que precisam consultar procedimentos de diagnóstico de rede, aplicações e segurança de acesso.

## Base de conhecimento (documentos reais)

Todos os documentos são PDFs originais, baixados das fontes oficiais, e ficam na pasta `docs/`.

| # | Arquivo em `docs/` | Documento | Fonte | Categoria |
|---|---|---|---|---|
| 1 | `01_cisco_troubleshooting_tcp_ip.pdf` | Troubleshooting TCP/IP (*Internetworking Troubleshooting Handbook*, cap. 7) | Cisco — https://www.cisco.com/en/US/docs/internetworking/troubleshooting/guide/tr1907.pdf | Rede |
| 2 | `02_lenovo_hardware_maintenance_manual.pdf` | Hardware Maintenance Manual (3ª edição, fev/2026) | Lenovo — https://download.lenovo.com/pccbbs/mobiles_pdf/x9_14_gen1_hmm_en.pdf | Hardware |
| 3 | `03_oracle_java_troubleshooting_guide.pdf` | Java SE 17 — Troubleshooting Guide | Oracle — https://docs.oracle.com/en/java/javase/17/troubleshoot/troubleshooting-guide.pdf | Software |
| 4 | `04_cisa_remote_access_software.pdf` | Guide to Securing Remote Access Software | CISA — https://www.cisa.gov/sites/default/files/2023-06/guide_to_securing_remote_access_software.pdf | Acesso |
| 5 | `05_cisa_phishing_resistant_mfa.pdf` | Implementing Phishing-Resistant MFA | CISA — https://www.cisa.gov/sites/default/files/publications/fact-sheet-implementing-phishing-resistant-mfa-508c.pdf | Acesso |

> Os documentos estão em inglês e as perguntas/respostas em português: o `nomic-embed-text` rende melhor em inglês, então perguntas muito vagas podem recuperar trechos menos precisos.

## Pré-requisitos

- **Ollama instalado e em execução na máquina** (https://ollama.com/download), com o modelo de embeddings baixado:

  ```bash
  ollama pull nomic-embed-text
  ```

  Os embeddings são gerados **localmente** (variável `OLLAMA_EMBED_BASE_URL`, padrão `http://localhost:11434`). Sem o Ollama rodando, a primeira execução falha ao gerar os embeddings.
- **Chave da Ollama Cloud** (`OLLAMA_API_KEY`) para o modelo de geração `gemma4:cloud`.
- **Acesso à internet na primeira execução** com o reranking ativo (padrão): o cross-encoder é baixado da Hugging Face (ver seção **Reranking**).

## Como executar (local — sem Colab)

```bash
cp .env.example .env          # edite com sua OLLAMA_API_KEY 
pip install -r requirements.txt
python -m rag.main            # chat no terminal
streamlit run app/interface.py  # interface web (opcional, ver seção "Interface web")
```

Na **primeira execução**, o `main.py` lê os PDFs de `docs/`, divide em chunks, gera os embeddings (`nomic-embed-text` local) e grava no ChromaDB (`chroma_db/`). Se o reranking estiver ativo, ele também baixa o cross-encoder da Hugging Face no primeiro uso (dependência `sentence-transformers`/`torch`). Nas execuções seguintes ele reaproveita a base.

Comandos dentro do chat: `fontes` (lista os documentos), `config` (troca a configuração de chunking), `filtro` (aplica/remove um filtro de metadata — ver seção **Metadata Filtering**), `rerank` (ativa/desativa o reranking em runtime — ver seção **Reranking**) e `sair`.

Para rodar a avaliação com RAGAS (gera o `ragas_resultado.md`):

```bash
python -m rag.ragas_eval
```

## Como adicionar novos documentos à base

1. Copie o novo arquivo **`.pdf`** (com texto selecionável, não escaneado) para a pasta `docs/`.
2. Registre o arquivo em `rag/metadata.py` (dicionário `METADADOS_POR_ARQUIVO`), informando `categoria`, `fornecedor` e `tipo_documento` reais. Sem esse registro o documento entra como `Não classificado` e não responde bem aos filtros.
3. **Apague a pasta `chroma_db/`** — o `main.py` só reconstrói a base quando ela não existe.
4. Rode `python -m rag.main`: a base é reconstruída com todos os PDFs de `docs/`.
5. Registre a fonte (título e link) na tabela acima.

Se um PDF não tiver texto extraível, o `loader.py` mostra um `[AVISO]` com o nome do arquivo.

## Exemplos de perguntas para testar o chatbot

Perguntas testadas no chatbot (`python -m rag.main`) que retornam resposta com informação da base, citando a fonte:

| Pergunta | Documento esperado |
|---|---|
| Como posso diagnosticar um problema de conectividade de rede usando ping e ipconfig? | Cisco — TCP/IP |
| O que é autenticação multifator resistente a phishing? | CISA — MFA |
| Quais são boas práticas para proteger o acesso remoto a sistemas corporativos? | CISA — Acesso remoto |
| Como solucionar problemas de uma aplicação Java que está travando? | Oracle — Java Troubleshooting |

**Perguntas que a base não cobre.** Nestes casos o chatbot responde que não encontrou a informação, em vez de inventar uma resposta:

- "Meu computador não liga, o que eu devo verificar primeiro?"
- "Se o notebook não liga nem com o cabo de energia conectado, o que pode ser o problema?"

Esse é o comportamento esperado do prompt (ver `rag/rag_chain.py`). A base cobre rede, segurança de acesso e depuração de aplicações Java, mas não traz um guia de diagnóstico básico de hardware: o manual da Lenovo é voltado a técnicos (substituição de peças e índice de sintomas).

## Diferencial CP2 — Metadata Filtering

Cada documento de `docs/` recebe metadados de classificação **reais** (extraídos do próprio conteúdo dos PDFs, ver tabela da base acima) durante o carregamento. Eles são preservados nos chunks e gravados no ChromaDB, permitindo filtrar a busca pelo recurso `where` do ChromaDB. O filtro é **opcional**: sem ele, a busca se comporta exatamente como antes.

Campos filtráveis (definidos em `rag/metadata.py`):

| Campo | Valores disponíveis |
|---|---|
| `categoria` | `Rede`, `Hardware`, `Software`, `Acesso` |
| `fornecedor` | `Cisco`, `Lenovo`, `Oracle`, `CISA` |
| `tipo_documento` | `Troubleshooting Guide`, `Hardware Maintenance Manual`, `Guide`, `Fact Sheet` |

### Como usar no chat

O comando `filtro` aplica um filtro a todas as perguntas seguintes (não é preciso repeti-lo a cada consulta):

```text
Você: filtro
  categoria: ['Acesso', 'Hardware', 'Rede', 'Software']
  fornecedor: ['CISA', 'Cisco', 'Lenovo', 'Oracle']
  tipo_documento: ['Fact Sheet', 'Guide', 'Hardware Maintenance Manual', 'Troubleshooting Guide']
Filtro atual: sem filtro
Filtro (ex: categoria=Rede, fornecedor=CISA) ou 'limpar': categoria=Rede
Filtro aplicado: categoria=Rede
```

Aceita mais de um campo separado por vírgula (ex.: `categoria=Acesso, fornecedor=CISA`). Use `filtro` → `limpar` para voltar à busca sem filtro.

### Como usar no código

```python
from rag.metadata import construir_filtro
from rag.rag_chain import buscar, buscar_chunks

# Sem filtro — comportamento original
buscar("Como diagnosticar conectividade de rede com ping?")

# Com filtro (recurso `where` do ChromaDB)
filtro = construir_filtro(categoria="Rede")
buscar("Como diagnosticar conectividade de rede com ping?", filtro=filtro)

# Só recuperar os trechos (sem gerar resposta), já com metadata preservado
docs = buscar_chunks("boas práticas de acesso remoto", filtro=filtro)
```

`construir_filtro()` aceita `categoria`, `fornecedor` e `tipo_documento`; campos vazios são ignorados e múltiplos campos geram um `{"$and": [...]}`. Um filtro sem correspondência retorna lista vazia (sem erro).

> Ao mudar os metadados, apague `chroma_db/` e rode `python -m rag.main` de novo para reindexar.

## Diferencial CP2 — Reranking

O pipeline ganhou uma etapa de **reranking** entre o retrieve e o generate (implementação em `rag/reranker.py`), sem substituir nenhum componente obrigatório:

```
ChromaDB (recupera fetch_k candidatos, com filtro opcional)
   → cross-encoder (reordena por relevância com a pergunta)
   → top_k finais → gemma4:cloud
```

- **Recuperação:** o ChromaDB continua sendo a primeira etapa, agora retornando **`RERANKER_FETCH_K` candidatos** (padrão 10) em vez dos 4 finais.
- **Reranker:** um **cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`, via `sentence-transformers`) pontua a relação de cada candidato com a pergunta e reordena a lista por score decrescente.
- **Generate:** apenas os **`RERANKER_TOP_K` melhores** (padrão 4) e seus metadados (fonte, página, categoria, fornecedor, tipo) vão para o prompt do `gemma4:cloud`.

**Dependências:** `torch` + `sentence-transformers` (o cross-encoder roda **localmente**, em CPU/GPU, e é **baixado da Hugging Face no primeiro uso** — cerca de 90 MB). A instalação do `torch` é pesada (vários GB no Linux, por causa das bibliotecas CUDA; costuma ser menor no Windows), então prefira instalar em um ambiente virtual novo: `pip install -r requirements.txt`. Se o cross-encoder não puder ser carregado (por exemplo, sem acesso à Hugging Face), o chat registra um `[AVISO]` e segue com a ordem original dos trechos.

### Como desativar/comparar

O reranking é opcional e pode ser desativado mantendo o comportamento original (busca `top_k` direto no ChromaDB):

1. **Por variável de ambiente** (padrão para todo o pipeline):
   ```bash
   $env:RERANKER_ATIVO="false"   # PowerShell
   # ou em .env: RERANKER_ATIVO=false
   ```
2. **No chat:** comando `rerank` → `off` (para reativar, `on`).
3. **No código:** parâmetro `rerank=`:
   ```python
   from rag.rag_chain import buscar, buscar_chunks

   buscar("pergunta")                           # usa RERANKER_ATIVO (env)
   buscar("pergunta", rerank=False)             # desativa só nessa chamada
   buscar_chunks("pergunta", k=4, rerank=True)  # recupera com reranking
   buscar_chunks("pergunta", k=4, rerank=False) # recuperação original
   ```

**Configuração (variáveis de ambiente, retêm os padrões acima):** `RERANKER_ATIVO` (`true`/`false`), `RERANKER_FETCH_K` (candidatos antes do rerank, padrão 10), `RERANKER_TOP_K` (trechos finais, padrão 4).

> **Nota de honestidade sobre ganhos:** o cross-encoder `ms-marco` foi treinado para o inglês e as perguntas da avaliação são em português; os scores absolutos são negativos e o reranking **reordena** os trechos, mas **não é garantia de ganho de qualidade** nas métricas. A avaliação RAGAS (`python -m rag.ragas_eval`) passou a coletar o contexto pelo mesmo caminho do generate (`buscar_chunks`), então rode com `RERANKER_ATIVO=false` para comparar com a baseline anterior registrada em `ragas_resultado.md`.

## Diferencial CP2 — Interface web (Streamlit)

O projeto ganhou uma interface gráfica web com **Streamlit** (`app/interface.py`). Ela **reutiliza o mesmo pipeline** do chat de terminal — em vez de chamar a chain diretamente, usa a função `buscar_com_fontes()` de `rag/rag_chain.py`, que devolve a resposta **e** os trechos que a sustentam — sem duplicar load, embeddings, retrieve ou generate.

> **Por que Streamlit e não Gradio?** O enunciado cita Gradio para esse diferencial, mas o grupo optou pelo Streamlit porque o Gradio estava apresentando erros no nosso ambiente. Já no CKP01 tivemos problemas de compatibilidade (o formato do histórico do `gr.Chatbot` mudou entre versões do Gradio e quebrou a interface) e, neste projeto, voltaram a aparecer erros ao usá-lo. Trocamos pelo Streamlit, que entrega a mesma funcionalidade pedida: chat com o RAG integrado e a fonte citada em cada resposta.

```bash
streamlit run app/interface.py   # abre em http://localhost:8501
```

**O que a interface oferece** (tudo na barra lateral, sem alterar o pipeline):

- **Histórico da conversa** mantido em `st.session_state`, com botão "Limpar conversa".
- **Configuração de chunking** (default: `CONFIG_PADRAO` de `rag/main.py`), alternando entre as coleções reindexadas do ChromaDB.
- **Filtro de metadata** (Metadata Filtering): selects de `categoria`, `fornecedor` e `tipo_documento` com os valores disponíveis, montados por `construir_filtro()` — com opção "Sem filtro".
- **Toggle de reranking** (Reranking): liga/desliga a etapa de cross-encoder ao vivo (default: `RERANKER_ATIVO`).
- **Fontes citadas:** cada resposta abre um expander **"Fontes usadas (N)"** com documento, página e classificação (categoria · fornecedor · tipo) por trecho.
- **Tratamento de erros amigável** (sem expor credenciais): se o Ollama local/Cloud estiver indisponível ou a base não tiver trechos suficientes, a interface informa claramente em vez de "inventar".

A funcionalidade foi validada com `streamlit.testing` (AppTest): render da página sem exceção, pergunta real respondida com fonte citada e filtro `categoria=Acesso` aplicado pela sidebar retornando apenas fontes dessa categoria.

> Requisito: `streamlit` entra no `requirements.txt` a partir desta atualização. A primeira execução do chat/interface baixa o cross-encoder da Hugging Face se o reranking estiver ativo.

## Arquitetura do pipeline

| Etapa | Arquivo | O que faz |
|---|---|---|
| load | `rag/loader.py` | Lê os PDFs de `docs/` (uma página = um `Document`, com `source`, `page` e os metadados de classificação) |
| metadata | `rag/metadata.py` | Registra `categoria`/`fornecedor`/`tipo_documento` de cada PDF e monta filtros no formato `where` do ChromaDB |
| split | `rag/splitter.py` | `RecursiveCharacterTextSplitter` com separadores `["\n\n", "\n", ". ", " ", ""]` e overlap de 15% |
| embed + store | `rag/embeddings_store.py` | `nomic-embed-text` (Ollama local) + ChromaDB, uma coleção por configuração de chunking |
| rerank | `rag/reranker.py` | Cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` reordena os `fetch_k` candidatos e devolve os `top_k` ao LLM (desativável) |
| retrieve + generate | `rag/rag_chain.py` | Chain única `retriever \| prompt \| llm \| parser` com `gemma4:cloud` (`temperature=0`), com suporte a filtro de metadata e reranking |
| avaliação | `rag/ragas_eval.py` | RAGAS (`faithfulness` + `answer_relevancy`) para cada configuração de chunking |
| interface | `rag/main.py` | CLI de perguntas e respostas com fonte citada |
| interface web | `app/interface.py` | UI Streamlit que reutiliza `buscar_com_fontes()` com filtro de metadata, reranking e fontes na sidebar |

A função `buscar(consulta)` (em `rag/rag_chain.py`) recebe uma string e devolve uma string, para ser reaproveitada como `@tool` no CKP03. Ela ganhou um parâmetro opcional `filtro=` (Metadata Filtering) — sem ele, a assinatura e o comportamento originais são mantidos.

## Comparação de chunking + RAGAS

Foram comparadas duas configurações de `chunk_size`, ambas com `chunk_overlap` de 15% e o mesmo `RecursiveCharacterTextSplitter`. Cada configuração tem sua própria coleção no ChromaDB. A avaliação usou 6 perguntas de teste (definidas em `rag/ragas_eval.py`), recuperação de `k = 4` trechos, `gemma4:cloud` com `temperature = 0` e embeddings `nomic-embed-text`. O gerador também atuou como juiz do RAGAS. O resultado completo está em `ragas_resultado.md`.

> **Importante:** os números abaixo foram medidos **antes** dos diferenciais da CP2 — sem reranking e sem filtro de metadata (equivalente a `RERANKER_ATIVO=false`). Hoje o reranking vem ligado por padrão; ao rodar `python -m rag.ragas_eval` de novo, o contexto passa pelo mesmo caminho do generate (`buscar_chunks`) e os valores podem mudar. Para reproduzir esta tabela, rode com `RERANKER_ATIVO=false`.

| Configuração | chunk_size | chunk_overlap | Faithfulness médio | Answer Relevancy médio |
|---|---|---|---|---|
| `pequeno_512` | 512 | 77 | **0,847** | **0,406** |
| `grande_1024` | 1024 | 154 | 0,735 | 0,152 |

**Resultado por pergunta** (a ordem é a de `PERGUNTAS_TESTE`; F = faithfulness, R = answer relevancy):

| # | Pergunta | F 512 | R 512 | F 1024 | R 1024 |
|---|---|---|---|---|---|
| 1 | Meu computador não liga, o que eu devo verificar primeiro? | 1,000 | 0,000 | 1,000 | 0,000 |
| 2 | Como posso diagnosticar um problema de conectividade de rede usando ping e ipconfig? | 0,900 | 0,871 | 1,000 | 0,000 |
| 3 | O que é autenticação multifator resistente a phishing? | 0,667 | 0,846 | 0,909 | 0,913 |
| 4 | Quais são boas práticas para proteger o acesso remoto a sistemas corporativos? | 0,714 | 0,717 | 0,000 | 0,000 |
| 5 | Como solucionar problemas de uma aplicação Java que está travando? | 0,800 | 0,000 | 1,000 | 0,000 |
| 6 | Se o notebook não liga nem com o cabo de energia conectado, o que pode ser o problema? | 1,000 | 0,000 | 0,500 | 0,000 |

**Configuração escolhida: `pequeno_512`.** O chatbot (`rag/main.py`) e a função `buscar()` usam essa configuração por padrão.

**Por que a 512 (e o que os dados mostram):**

- **Faithfulness:** as duas configurações passam da meta de aprovação (0,7), mas só a 512 fica bem acima dela (0,847 contra 0,735). Nenhuma chega à zona ideal (0,9).
- **Answer relevancy:** é a diferença mais clara. A 512 teve relevância maior que zero em 3 das 6 perguntas; a 1024, em 1 de 6.
- **O que sustenta a média da 512:** a vantagem em faithfulness vem principalmente da pergunta 4, em que a 1024 teve 0,000 (contra 0,714). Em 3 das 6 perguntas (2, 3 e 5), a 1024 teve faithfulness igual ou maior. Por isso a escolha se apoia mais na relevância e na média do que numa vantagem uniforme.
- **Hipótese:** com `k = 4`, a 1024 entrega ao modelo até ~4.000 caracteres de contexto (contra ~2.000 na 512), com mais texto pouco relacionado à pergunta misturado em cada trecho. Trechos menores tendem a ser mais focados.

**Limitações da avaliação:**

- São apenas 6 perguntas e uma execução por configuração. As métricas do RAGAS são calculadas por um LLM e variam entre execuções, então diferenças pequenas devem ser lidas com cautela.
- **Perguntas de hardware (1 e 6):** em teste manual no chatbot, as duas retornaram "não encontrei a informação na base". É o comportamento esperado do prompt para perguntas sem cobertura e é a explicação provável para a answer relevancy 0,000 nelas, nas duas configurações: o RAGAS atribui 0 a respostas evasivas. O manual da Lenovo é voltado a técnicos (substituição de peças e índice de sintomas) e não traz passos diretos de diagnóstico para "o computador não liga".
- **Outros zeros não explicados:** também houve answer relevancy 0,000 em perguntas que o chatbot respondeu normalmente no teste manual (como a 5, nas duas configurações). A causa não foi investigada; pode estar em trechos da resposta que o juiz considera evasivos ou em falhas do juiz (o mesmo `gemma4:cloud`) ao interpretar a saída. Por isso a métrica de relevância deve ser lida com cautela.

## Notas técnicas

- **Versão do ragas:** fixada em `0.3.3` no `requirements.txt` de propósito. A partir da `0.3.4` ele exige o pacote `scikit-network`, que não tem build pronta para Python recente no Windows e quebra o `pip install`. A `0.3.3` funciona com o código do projeto.

## CKP03 — Agente de triagem de TI (ReAct + AgentExecutor)

A CP3 transforma o pipeline RAG em uma *tool* de um agente inteligente, mantendo as ENTREGAS da CP1 e CP2 intactas (`rag/` e `app/` sem mudanças de comportamento).

**Tecnologias da CP3:** Python 3.14 (venv do projeto), LangChain 1.x (`langchain_core`) + `langchain-classic 1.0.x` (`create_react_agent` + `AgentExecutor` — motivo do uso da *classic* documentado abaixo), `gemma4:cloud` via Ollama Cloud para o LLM do agente e da tool RAG, `nomic-embed-text` (Ollama local) + ChromaDB para o RAG, Gradio 6.x para a interface web e `.env`/`python-dotenv` para a chave da Ollama Cloud.

- **Modelo de chat:** exclusivamente `gemma4:cloud` (`app/chain.py` / `rag/rag_chain.py`), via Ollama Cloud, com `OLLAMA_API_KEY` carregada de variável de ambiente (`.env`), nunca hardcoded.
- **Requisitos obrigatórios atendidos:** agente ReAct criado com `create_react_agent` e executado pelo `AgentExecutor` (com `handle_parsing_errors`, `max_iterations` e `max_execution_time`); pipeline RAG da CP2 exposto como única tool de consulta apelidada `buscar(consulta)` (input único de string, formato exigido pelo parser ReAct, em `agent/tools.py`); chat exclusivo via `gemma4:cloud`; interface web Gradio com histórico por sessão e fontes citadas; três cenários validados (RAG, outra ferramenta, sem ferramenta — seção de interações reais).
- **Arquitetura:** `langchain_classic.agents.create_react_agent` + `AgentExecutor` (ReAct). Por que `langchain_classic` e não `langchain.agents`: no ambiente deste projeto (Python 3.14 + Pydantic) o módulo legado `langchain.chains` quebra na avaliação de type hints (issue langchain-ai/langchain#33449, documentada em `app/memory_manager.py`) — a `langchain-classic` funciona e está fixada no `requirements.txt`.
- **Domínio:** triagem e suporte técnico de TI corporativo (mesmo domínio das CP1/CP2).
- **Guardrail de entrada (bônus):** antes de qualquer ferramenta rodar, `agent/guardrail.py`
  valida a mensagem de forma heurística (regex, sem LLM e sem serviços externos): entradas
  vazias, acima de 2.000 caracteres, tentativas de prompt injection ("ignore as instruções
  anteriores", extração de system prompt/jailbreak) e assuntos claramente fora do domínio de
  TI são recusadas educadamente. Para não bloquear perguntas legítimas, a regra de domínio
  só recusa quando há marcador off-topic claro e nenhum sinal de TI — e a recusa não entra
  no histórico do agente. A validação vive na porta única do agente
  (`AgenteTriagemTI.enviar_mensagem_com_passos`, `agent/agent.py`), por isso CLI e interface
  Gradio herdam o guardrail sem duplicação; testes em `tests/test_guardrail_cp3.py`.
- **Compressão de contexto (bônus), em DOIS níveis distintos:** (1) histórico de conversa entre turnos reutilizando `aplicar_token_buffer`/`trim_messages` da CP1; (2) dentro de um mesmo turno, `agent/scratchpad_compressor.py` comprime o **scratchpad ReAct** — quando as observações acumuladas das ferramentas passam do limite (`SCRATCHPAD_LIMITE_CARACTERES`, padrão 3.000 chars), as antigas viram UMA síntese extrativa determinística (sem LLM extra) que mantém cabeças técnicas e os blocos `Fontes:` verbatim (nada de citação alucinada), e apenas a janela das mais recentes (`SCRATCHPAD_JANELA_RECENTES`, padrão 2) segue intacta; a subclasse `AgentExecutorComCompressao` injeta isso em cada iteração do loop do `AgentExecutor` sem forkar nada (só reescreve `_take_next_step`). Isso NÃO é limitar o histórico visual do Gradio; na interface, cada sessão do navegador tem agente próprio (sem misturar conversas).
- **Guardrail (comportamental):** relatos insuficientes são tratados pela tool `verificar_informacoes_minimas` e pelas regras dos prompts (sem inventar dados e sem registrar chamados em sistemas externos); o filtro de domínio da conversa é feito pelo guardrail de entrada acima.

### Ferramentas do agente (4, em `agent/tools.py`)

| # | Ferramenta | Reuso | Quando usar |
|---|---|---|---|
| 1 | `consultar_base_conhecimento_ti` (`@tool`) | Pipeline RAG da CP2 inteiro: busca UMA vez (ChromaDB + reranking), geração com o mesmo `RAG_PROMPT`/`gemma4:cloud`, resposta + seção `Fontes:` com documento/página/classificação reais. Se a base não estiver indexada, devolve as instruções de indexação da CP2 | **Dúvidas técnicas e perguntas de "como fazer"** ("como diagnosticar...", "o que recomenda para...") — qualquer coisa resolvida pelos manuais em `docs/` |
| 2 | `classificar_chamado_ti` | Saída Pydantic com categoria e prioridade (enums da CP1) e critérios explícitos de impacto/urgência | **Relato de problema com contexto suficiente** — para definir categoria (hardware/software/rede/acesso) e prioridade sugerida antes de escalar |
| 3 | `gerar_checklist_diagnostico` | Checklist seguro/ordenado; proíbe ações destrutivas e exposição de credenciais | **Autoatendimento do usuário** — quando ele quer tentar resolver antes de escalar; etapas seguras e em ordem |
| 4 | `verificar_informacoes_minimas` | Valida completude do relato; não consulta sistemas externos | **Relato vago/incompleto** ("não funciona", falta SO/erro/equipamento) — lista exatamente o que falta antes de classificar |

O agente Responde SEM ferramenta em cumprimentos, perguntas sobre seu próprio papel e despedidas (prompts orientam o Final Answer direto nesses casos).

### Como executar

```bash
pip install -r requirements.txt
python -m agent.main               # conversa no terminal (comandos: 'limpar' e 'sair')
python -m agent.interface_gradio   # interface web (Gradio) com share=True
python -m unittest discover -s tests   # 35 testes offline (agente, guardrail, compressão) — não exige Ollama
```

Pré-requisitos das execuções acima: `OLLAMA_API_KEY` no `.env`, Ollama Desktop em execução com `nomic-embed-text` (embeddings locais) e a base da CP2 indexada (`chroma_db/` presente — na primeira vez, rode `python -m rag.main`). Sem a base indexada, o agente não bloqueia: a tool RAG devolve as instruções de indexação em vez de inventar.

### Configuração (`.env` — Ollama Cloud e Ollama local)

Copie `.env.example` para `.env` e preencha:

| Variável | Para quê | Valor do projeto |
|---|---|---|
| `OLLAMA_API_KEY` | **Chave da Ollama Cloud**, usada pelo `gemma4:cloud` (obrigatória; obtida na plataforma Ollama) | sua chave pessoal |
| `OLLAMA_BASE_URL` | Endpoint da Ollama Cloud | `https://ollama.com` |
| `OLLAMA_EMBED_BASE_URL` | Ollama **local** (Desktop) para os embeddings | `http://localhost:11434` |

A chave **nunca sobe no `.zip` nem para o repositório** — só o `.env.example` é versionado. Bônus de compressão do scratchpad via ambiente (opcionais): `SCRATCHPAD_LIMITE_CARACTERES` (padrão 3000) e `SCRATCHPAD_JANELA_RECENTES` (padrão 2).

### URL pública (share=True)

O `launch(share=True)` cria uma URL temporária do tipo `https://xxxx.gradio.live`, mostrada logo que o app sobe:

- A URL **só funciona enquanto o processo estiver em execução** — ao encerrar o app (Ctrl+C ou fechar o terminal), ela deixa de responder; cada nova execução gera uma URL nova.
- Para publicar de novo: rode `python -m agent.interface_gradio` e **copie a URL nova do terminal**.
- Para confirmar o acesso de outro dispositivo (celular/outra máquina), abra a URL no navegador e verifique se aparece o título **"DocAgent de TI — CP3"**; na máquina de desenvolvimento, o app roda em `http://127.0.0.1:7866` (ou a porta impressa no terminal).

### Evidências da interface Gradio (validações reais durante o desenvolvimento)

A interface foi validada de duas formas, documentadas sem alteração do código:

1. **Do servidor de desenvolvimento:** elevar o app (`python -m agent.interface_gradio`) e conversar direto em `http://127.0.0.1:7866` — resposta com fontes no painel "Fontes consultadas nesta resposta", painel "Ferramentas usadas nesta resposta" apenas com os NOMES das ferramentas (o raciocínio ReAct interno nunca é exibido), botão "Limpar conversa" e os 3 exemplos clicáveis (`gr.Examples`).
2. **De outra máquina, via `gradio_client` na URL pública `*.gradio.live`:** sessões distintas mantiveram históricos independentes (pergunta de contexto em A não "vazou" para B), "Limpar conversa" zerou tela e histórico interno (a próxima resposta ignorou a pergunta anterior), e perguntas fora de domínio/injeção foram barradas pelo guardrail na própria tela, sem stack trace e sem expor credenciais.

> **Evidência visual (ação manual):** URLs `*.gradio.live` e screenshots são temporários — cada execução gera URL nova e encerrar o app derruba a antiga, então não há screenshot "de arquivo" que sobreviva à entrega. Para a evidência da entrega: levantar o app (`python -m agent.interface_gradio`), capturar a tela do navegador (aba com título "DocAgent de TI — CP3", painéis de fontes/ferramentas e uma conversa real) e guardar o print junto ao ZIP.

### Interações reais registradas (22/10/2026, gemma4:cloud — ambiente do projeto)

1. **Interação com RAG — pergunta:** "O que a base de conhecimento recomenda sobre autenticação anti-phishing e MFA?" → **ferramentas usadas:** `consultar_base_conhecimento_ti`. **Resultado:** resposta correta sobre phishing-resistant MFA (FIDO/WebAuthn, PKI, imunidade a SIM swap/push bombing) encerrada com "**Fontes:** 05_cisa_phishing_resistant_mfa.pdf (páginas 0, 1 e 2) — Acesso · CISA · Fact Sheet".
2. **Interação com outra ferramenta — pergunta:** "O usuário só relatou 'o computador está lento às vezes e o ponto de rede cai'. Gere um checklist de diagnóstico com etapas técnicas." → **ferramentas usadas:** `gerar_checklist_diagnostico`. **Resultado:** checklist ordenado (verificação física do cabo, reinicialização, teste de navegação em dois sites, Gerenciador de Tarefas) + perguntas de esclarecimento (SO, quando cai, mensagens de erro).
3. **Interação sem ferramenta — pergunta:** "Obrigado pela explicação! Já entendi como usar. Até mais!" → **ferramentas usadas:** nenhuma (resposta direta). **Resultado:** despedida cordial sem chamar ferramenta alguma ("Por nada! Fico feliz em ter ajudado..."), chegando direto ao Final Answer na 1ª iteração.

Validação adicional na mesma sessão: classificação ("o colaborador em home office não consegue imprimir; urgência alta e impacto médio") → `classificar_chamado_ti` → Hardware/Média/Impressora + perguntas pendentes; consultas RAG de TCP/IP citaram `01_cisco_troubleshooting_tcp_ip.pdf`, página 17; perguntas fora de domínio ("receita de bolo", "jogo do Grêmio") e uma tentativa de prompt injection foram barradas pelo guardrail de entrada antes de qualquer ferramenta; a compressão do scratchpad engatou em 2 iterações reais do executor (fluxo com 3 ferramentas) quando as observações passaram do limite configurado.

### Exemplos práticos de perguntas

1. **Agente usa o RAG:** "O que é autenticação multifator resistente a phishing?" → o agente escolhe `consultar_base_conhecimento_ti` e responde com base no `05_cisa_phishing_resistant_mfa.pdf`, citando páginas e a classificação (Acesso · CISA · Fact Sheet).
2. **Agente usa outra ferramenta:** "Meu notebook não liga e preciso dele hoje para uma reunião importante." → escolhe `verificar_informacoes_minimas`, depois `classificar_chamado_ti` (hardware/alta) e por fim `gerar_checklist_diagnostico`, sem afirmar que registrou chamado.
3. **Agente responde sem ferramenta:** "Olá, tudo bem? Quem é você e o que consegue fazer?" → Final Answer direto, sem chamadas (painel da interface mostra "Nenhuma — o agente respondeu diretamente").

### Erros conhecidos e observações (nenhum bloqueia a entrega)

- **Erros de parsing ReAct absorvidos:** o `gemma4:cloud` ocasionalmente emite uma iteração fora do formato Thought/Action/Action Input; o `AgentExecutor` está com `handle_parsing_errors=True`, o executor insere um passo interno `_Exception` e o agente consegue se recuperar — o filtro de whitelist das ferramentas impede que esse passo apareça como "ferramenta usada" na interface. Não é um bug do código: é comportamento do modelo, tratado na execução.
- **Latência:** cada chamada ao `gemma4:cloud` leva tipicamente 30–90 s; um turno com várias ferramentas pode demorar alguns minutos. Ajuste de `MAX_ITERACOES`/`MAX_TEMPO_SEGUNDOS` está em `agent/agent.py`.
- **URL pública temporária:** `*.gradio.live` vale só enquanto o app roda (detalhes na seção "Evidências da interface Gradio").
- **Nota histórica (CP2):** na CP2 a interface web foi feita em Streamlit porque o Gradio apresentava erros no ambiente da época (detalhe na seção "Diferencial CP2 — Interface web (Streamlit)"); na CP3 o Gradio 6.30 passou a funcionar e é a interface oficial do agente.
- **Interface de apresentação apenas:** o Gradio não replica lógica — RAG, guardrail e compressão vivem em `agent/` (`agent/agent.py`, `agent/tools.py`, `agent/guardrail.py`, `agent/scratchpad_compressor.py`), reaproveitados tanto pela CLI (`python -m agent.main`) quanto pela web (interface).
