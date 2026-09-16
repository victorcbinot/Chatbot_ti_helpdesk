"""
main.py — Entry point do projeto (python -m app.main).
"""
from app.chain import (
    build_conversation_chain,
    build_extraction_chain,
    formatar_historico_para_extracao,
)


def imprimir_analise(analise) -> None:

    print("\n" + "=" * 60)
    print("ANÁLISE ESTRUTURADA DO CHAMADO")
    print("=" * 60)
    print(f"Categoria:            {analise.categoria.value}")
    print(f"Urgência:             {analise.urgencia.value}")
    print(f"Sistema afetado:      {analise.sistema_afetado}")
    print(f"Resumo:               {analise.resumo_problema}")
    print(f"Ação recomendada:     {analise.acao_recomendada}")
    print(
        "Requer escalonamento: "
        f"{'Sim' if analise.requer_escalonamento else 'Não'}"
    )
    print("=" * 60 + "\n")


def main() -> None:
    conversation_chain = build_conversation_chain()
    extraction_chain = build_extraction_chain()


    historico_tuplas: list[tuple[str, str]] = []

    print("=" * 60)
    print("TI.Assist — Assistente de Triagem de Chamados de TI")
    print("=" * 60)
    print("Descreva seu problema técnico. Comandos especiais:")
    print("  analisar  -> gera a análise estruturada do chamado")
    print("  sair      -> encerra a conversa")
    print("=" * 60 + "\n")

    while True:
        mensagem = input("Você: ").strip()

        if not mensagem:
            continue

        if mensagem.lower() in ("sair", "exit", "quit"):
            print("\nTI.Assist: Até mais! Chamado encerrado.")
            break

        if mensagem.lower() == "analisar":
            if not historico_tuplas:
                print(
                    "\nTI.Assist: Ainda não há conversa suficiente para "
                    "analisar. Descreva o problema primeiro.\n"
                )
                continue

            conversa_formatada = formatar_historico_para_extracao(historico_tuplas)
            try:
                analise = extraction_chain.invoke({"conversa": conversa_formatada})
                imprimir_analise(analise)
            except Exception as exc:
                print(f"\nErro ao gerar análise estruturada: {exc}\n")
            continue

        resposta = conversation_chain.enviar_mensagem(mensagem)
        print(f"\nTI.Assist: {resposta}\n")

        historico_tuplas.append(("user", mensagem))
        historico_tuplas.append(("assistant", resposta))


if __name__ == "__main__":
    main()