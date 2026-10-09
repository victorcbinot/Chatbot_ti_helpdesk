# CKP02 — DocMind RAG · Assistente de Triagem de Chamados de TI

**Prompt Engineering & AI · FIAP · 2º Semestre 2026**
**Integrantes:**
| Nome | RM |
|-|-|
| Gustavo Kunitaki | 571400 |
| Pedro Ferreras | 568713 |
| Pedro Santos | 571017 |
| Victor Binot | 571499 |**

## Domínio

**Domínio:** triagem de chamados de suporte técnico de TI (helpdesk corporativo) — o mesmo do CKP01.

**Por que foi escolhido:** é um domínio com documentação técnica real, pública e confiável (fabricantes, órgãos de segurança), o que permite montar uma base de conhecimento verdadeira. No CKP01 o chatbot coletava o relato do problema; aqui ele passa a **responder com base em documentos técnicos reais**, citando a fonte. No CKP03 o pipeline vira uma *tool* (`buscar(consulta)`) de um agente.

**Usuários-alvo:** colaboradores e analistas de suporte de uma empresa que precisam consultar procedimentos de diagnóstico de rede, aplicações e segurança de acesso.

## Base de conhecimento (documentos reais)

Todos os documentos são PDFs originais, baixados das fontes oficiais, e ficam na pasta `docs/`.

| # | Arquivo em `docs/` | Documento | Fonte | Categoria |
|---|---|---|---|---|
| 1 | `01_cisco_troubleshooting_tcp_ip.pdf` | Troubleshooting TCP/IP (*Internetworking Troubleshooting Handbook*, cap. 7) | Cisco — https://www.cisco.com/en/US/docs/internetworking/troubleshooting/guide/tr1907.pdf | Rede |
| 2 | `02_lenovo_hardware_maintenance_manual.pdf` | ThinkPad X9-14 Gen 1 — Hardware Maintenance Manual (3ª edição, fev/2026) | Lenovo — https://download.lenovo.com/pccbbs/mobiles_pdf/x9_14_gen1_hmm_en.pdf | Hardware |
| 3 | `03_oracle_java_troubleshooting_guide.pdf` | Java SE 17 — Troubleshooting Guide | Oracle — https://docs.oracle.com/en/java/javase/17/troubleshoot/troubleshooting-guide.pdf | Software |
| 4 | `04_cisa_remote_access_software.pdf` | Guide to Securing Remote Access Software | CISA — https://www.cisa.gov/sites/default/files/2023-06/guide_to_securing_remote_access_software.pdf | Acesso |
| 5 | `05_cisa_phishing_resistant_mfa.pdf` | Implementing Phishing-Resistant MFA | CISA — https://www.cisa.gov/sites/default/files/publications/fact-sheet-implementing-phishing-resistant-mfa-508c.pdf | Acesso |

> Os documentos estão em inglês e as perguntas/respostas em português: o `nomic-embed-text` rende melhor em inglês, então perguntas muito vagas podem recuperar trechos menos precisos.

## Como executar (local — sem Colab)

```bash
cp .env.example .env          # edite com sua OLLAMA_API_KEY — este arquivo NÃO vai no .zip
pip install -r requirements.txt
python -m rag.main            # chat no terminal
```

Na **primeira execução**, o `main.py` lê os PDFs de `docs/`, divide em chunks, gera os embeddings (`nomic-embed-text`) e grava no ChromaDB (`chroma_db/`). Nas execuções seguintes ele reaproveita essa base.

Comandos dentro do chat: `fontes` (lista os documentos), `config` (troca a configuração de chunking) e `sair`.

Para rodar a avaliação com RAGAS (gera o `ragas_resultado.md`):

```bash
python -m rag.ragas_eval
```

## Como adicionar novos documentos à base

1. Copie o novo arquivo **`.pdf`** (com texto selecionável, não escaneado) para a pasta `docs/`.
2. **Apague a pasta `chroma_db/`** — o `main.py` só reconstrói a base quando ela não existe.
3. Rode `python -m rag.main`: a base é reconstruída com todos os PDFs de `docs/`.
4. Registre a fonte (título e link) na tabela acima.

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

## Arquitetura do pipeline

| Etapa | Arquivo | O que faz |
|---|---|---|
| load | `rag/loader.py` | Lê os PDFs de `docs/` (uma página = um `Document`, com `source` e `page`) |
| split | `rag/splitter.py` | `RecursiveCharacterTextSplitter` com separadores `["\n\n", "\n", ". ", " ", ""]` e overlap de 15% |
| embed + store | `rag/embeddings_store.py` | `nomic-embed-text` (Ollama Cloud) + ChromaDB, uma coleção por configuração de chunking |
| retrieve + generate | `rag/rag_chain.py` | Chain única `retriever \| prompt \| llm \| parser` com `gemma4:cloud` (`temperature=0`) |
| avaliação | `rag/ragas_eval.py` | RAGAS (`faithfulness` + `answer_relevancy`) para cada configuração de chunking |
| interface | `rag/main.py` | CLI de perguntas e respostas com fonte citada |

A função `buscar(consulta)` (em `rag/rag_chain.py`) recebe uma string e devolve uma string, para ser reaproveitada como `@tool` no CKP03.

## Comparação de chunking + RAGAS

Foram comparadas duas configurações de `chunk_size`, ambas com `chunk_overlap` de 15% e o mesmo `RecursiveCharacterTextSplitter`. Cada configuração tem sua própria coleção no ChromaDB. A avaliação usou 6 perguntas de teste (definidas em `rag/ragas_eval.py`), recuperação de `k = 4` trechos, `gemma4:cloud` com `temperature = 0` e embeddings `nomic-embed-text`. O gerador também atuou como juiz do RAGAS. O resultado completo está em `ragas_resultado.md`.

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
- **Chave de API:** `OLLAMA_API_KEY` fica no `.env` (carregado com `python-dotenv`) e nunca vai no `.zip`; só o `.env.example`.
