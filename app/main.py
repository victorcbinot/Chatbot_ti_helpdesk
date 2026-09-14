"""
main.py — Entry point do projeto (python -m app.main).
"""

import gradio as gr

from app.chain import build_conversation_chain, build_extraction_chain, formatar_historico_para_extracao

conversation_chain = build_conversation_chain()
extraction_chain = build_extraction_chain()


def responder(mensagem: str, historico: list):
    """Callback do chat: envia a mensagem para a Chain 1 (com memória)."""
    resposta = conversation_chain.enviar_mensagem(mensagem)
    historico = historico + [(mensagem, resposta)]
    return "", historico


def analisar_chamado(historico: list):
    """
    Callback do botão "Analisar chamado": roda a Chain 2 sobre o histórico
    atual da conversa e retorna a saída estruturada (AnaliseChamado) já
    validada pelo Pydantic.
    """
    if not historico:
        return "Converse com o assistente antes de gerar a análise."

    mensagens = []
    for usuario_msg, assistente_msg in historico:
        mensagens.append(("user", usuario_msg))
        mensagens.append(("assistant", assistente_msg))

    conversa_formatada = formatar_historico_para_extracao(mensagens)

    try:
        analise = extraction_chain.invoke({"conversa": conversa_formatada})
    except Exception as exc:
        return f"Erro ao gerar análise estruturada: {exc}"

    return (
        f"**Categoria:** {analise.categoria.value}\n"
        f"**Urgência:** {analise.urgencia.value}\n"
        f"**Sistema afetado:** {analise.sistema_afetado}\n"
        f"**Resumo:** {analise.resumo_problema}\n"
        f"**Ação recomendada:** {analise.acao_recomendada}\n"
        f"**Requer escalonamento humano:** "
        f"{'Sim' if analise.requer_escalonamento else 'Não'}"
    )


def construir_interface() -> gr.Blocks:
    with gr.Blocks(title="TI.Assist — Triagem de Chamados de TI") as demo:
        gr.Markdown("# 🖥️ TI.Assist — Assistente de Triagem de Chamados de TI")
        gr.Markdown(
            "Converse descrevendo seu problema técnico. Quando achar que já "
            "deu detalhes suficientes, clique em **Analisar chamado** para "
            "gerar a triagem estruturada."
        )

        chatbot = gr.Chatbot(label="Conversa com o TI.Assist", height=420)
        entrada = gr.Textbox(
            label="Sua mensagem", placeholder="Ex: meu notebook não liga..."
        )
        enviar_btn = gr.Button("Enviar", variant="primary")

        analisar_btn = gr.Button("📋 Analisar chamado (saída estruturada)")
        saida_analise = gr.Markdown(label="Análise do chamado")

        enviar_btn.click(responder, [entrada, chatbot], [entrada, chatbot])
        entrada.submit(responder, [entrada, chatbot], [entrada, chatbot])
        analisar_btn.click(analisar_chamado, [chatbot], [saida_analise])

    return demo


if __name__ == "__main__":
    interface = construir_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860)