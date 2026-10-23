"""
main.py — CLI de demonstração/teste do agente da CP3 (python -m agent.main).

Loop simples no terminal que usa AgenteTriagemTI (agent/agent.py). Serve
para validar manualmente os 3 cenários obrigatórios do enunciado:
  (a) pergunta técnica que usa o RAG (consultar_base_conhecimento_ti);
  (b) relato de problema que usa outra ferramenta
      (verificar_informacoes_minimas, classificar_chamado_ti,
      gerar_checklist_diagnostico);
  (c) cumprimento/pergunta geral respondida SEM ferramenta.

Comandos: 'limpar' (zera o histórico), 'sair' (encerra).
Na abertura, verifica se a base de conhecimento da CP2 está indexada e —
se não estiver — imprime as instruções de indexação (sem bloquear o chat).
"""

from agent.agent import AgenteTriagemTI
from agent.tools import base_indexada, INSTRUCAO_INDEXACAO


def main() -> None:
    print("=" * 60)
    print("TI.Assist (CP3) — Agente de Triagem de TI com ReAct")
    print("=" * 60)

    if not base_indexada():
        print(f"[AVISO] {INSTRUCAO_INDEXACAO}\n")

    agente = AgenteTriagemTI()
    print("Comandos: 'limpar' (zera histórico), 'sair'.\n")

    while True:
        try:
            mensagem = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTI.Assist: Até mais!")
            break

        if not mensagem:
            continue
        if mensagem.lower() in ("sair", "exit", "quit"):
            print("\nTI.Assist: Até mais!")
            break
        if mensagem.lower() == "limpar":
            agente.limpar_conversa()
            print("(histórico da conversa limpo)\n")
            continue

        resposta = agente.enviar_mensagem(mensagem)
        print(f"\nTI.Assist: {resposta}\n")


if __name__ == "__main__":
    main()
