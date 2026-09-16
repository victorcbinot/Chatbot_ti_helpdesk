"""
meta_prompting.py — Diferencial: Meta Prompting (Aula 04).

Usa o próprio modelo (gemma4:cloud) para revisar criticamente o system
prompt do chat (SYSTEM_PROMPT_CHAT) e propor uma versão melhorada,
documentando o antes/depois

Execução:
    python -m app.meta_prompting
"""

from langchain_core.messages import HumanMessage, SystemMessage

from app.chain import build_llm
from app.prompts import SYSTEM_PROMPT_CHAT

META_PROMPT = """
Você é um especialista em prompt engineering, especializado em revisar
system prompts de chatbots de atendimento.

Você vai receber o system prompt de um chatbot de TRIAGEM DE CHAMADOS DE
TI (helpdesk corporativo). Sua tarefa:

1. Aponte de 3 a 5 pontos fracos objetivos do prompt atual (ambiguidades,
   regras que podem conflitar entre si, falta de exemplos, instruções
   vagas, casos de borda não cobertos, etc.).
2. Reescreva uma VERSÃO MELHORADA do prompt, corrigindo esses pontos.
   Mantenha a mesma estrutura de tags XML (<persona>, <dominio>, <regras>,
   <restricoes>) usada no original - não mude o domínio (continua sendo
   triagem de chamados de TI).

Responda estritamente neste formato, sem texto antes ou depois:

## Pontos de melhoria identificados
1. ...
2. ...
3. ...

## Prompt melhorado

<persona>
...
</persona>

<dominio>
...
</dominio>

<regras>
...
</regras>

<restricoes>
...
</restricoes>
""".strip()


def rodar_meta_prompting() -> str:

    llm = build_llm(temperature=0.2)
    resposta = llm.invoke(
        [
            SystemMessage(content=META_PROMPT),
            HumanMessage(
                content=f"System prompt atual do chatbot:\n\n{SYSTEM_PROMPT_CHAT}"
            ),
        ]
    )
    return resposta.content


def salvar_resultado(
    resultado_do_modelo: str, caminho: str = "meta_prompting_resultado.md"
) -> None:

    conteudo = f"""# Meta Prompting - Antes e Depois

Diferencial da Aula 04: usamos o próprio modelo (gemma4:cloud) para
revisar e melhorar o system prompt do chat.

## System prompt ANTES (original, usado em prompts.py)

```
{SYSTEM_PROMPT_CHAT}
```

## Resultado da revisão feita pelo próprio modelo

{resultado_do_modelo}
"""
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(conteudo)
    print(f"Resultado salvo em: {caminho}")
    print("Copie o conteúdo relevante para a seção 'Meta prompting' do README.md.")


if __name__ == "__main__":
    print("Rodando meta prompting sobre o system prompt do chat (SYSTEM_PROMPT_CHAT)...\n")
    resultado = rodar_meta_prompting()
    print(resultado)
    print()
    salvar_resultado(resultado)