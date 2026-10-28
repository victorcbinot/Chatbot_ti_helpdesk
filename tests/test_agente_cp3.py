"""
test_agente_cp3.py — Testes OFFLINE do núcleo do agente da CP3.

Não tocam na rede: o LLM é substituído por FakeListChatModel (langchain-core)
e a recuperação RAG é simulada com stub de buscar_com_fontes. Validam:
  - as 4 ferramentas existem, com nome/descrição e input único (ReAct);
  - o prompt ReAct tem todas as variáveis exigidas pelo create_react_agent;
  - a tool RAG chama o pipeline da CP2 (buscar_com_fontes) UMA vez e
    responde preservando "Fontes:";
  - as tools estruturadas retornam o formato acordado;
  - o AgentExecutor (create_react_agent) seleciona: (a) tool RAG e
    (b) nenhuma ferramenta em cenários apropriados.

Execução:   python -m unittest discover -s tests -v   (na raiz do projeto)
Requisitos: venv com requirements.txt instalado. NÃO exige Ollama
            local/cloud nem OLLAMA_API_KEY (tudo simulado).
"""

import unittest
from unittest.mock import patch

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

# Módulo inteiro para patch nos pontos referenciados pelas funções das tools.
import agent.tools as tools_mod
from agent.agent import MAX_ITERACOES
from agent.prompts import REACT_PROMPT
from agent.tools import FERRAMENTAS


def _doc_fonte(texto: str) -> Document:
    return Document(
        page_content=texto,
        metadata={
            "source": "01_cisco_troubleshooting_tcp_ip.pdf",
            "page": 12,
            "categoria": "Rede",
            "fornecedor": "Cisco",
            "tipo_documento": "Troubleshooting Guide",
        },
    )


class TestFerramentasRegistradas(unittest.TestCase):

    def test_quatro_ferramentas_com_nomes(self):
        nomes = [t.name for t in FERRAMENTAS]
        self.assertEqual(len(FERRAMENTAS), 4)
        for nome in (
            "consultar_base_conhecimento_ti",
            "classificar_chamado_ti",
            "gerar_checklist_diagnostico",
            "verificar_informacoes_minimas",
        ):
            self.assertIn(nome, nomes)

    def test_input_unico_e_descricao(self):
        # O parser ReAct passa a Action Input como UMA string: cada tool
        # deve ter exatamente 1 parâmetro obrigatório e ter descrição.
        for t in FERRAMENTAS:
            self.assertEqual(len(t.args), 1, t.name)
            unico = next(iter(t.args.values()))
            self.assertTrue(unico.get("required", True), t.name)
            self.assertTrue(t.description.strip(), t.name)


class TestReactPrompt(unittest.TestCase):

    def test_variaveis_obrigatorias(self):
        variaveis = set(REACT_PROMPT.input_variables)
        for exigida in ("tools", "tool_names", "input", "agent_scratchpad"):
            self.assertIn(exigida, variaveis)
        # Histórico comprimido entra como variável OPICIONAL (placeholder).
        self.assertIn("chat_history", REACT_PROMPT.optional_variables)

    def test_renderiza_com_ferramentas_da_producao(self):
        lista_tools = "\n".join(f"- {t.name}" for t in FERRAMENTAS)
        mensagens = REACT_PROMPT.invoke(
            {
                "input": "pergunta de teste",
                "tools": lista_tools,
                "tool_names": ", ".join(t.name for t in FERRAMENTAS),
                "agent_scratchpad": "",
                "chat_history": [],
            }
        ).to_string()
        self.assertIn("pergunta de teste", mensagens)
        self.assertIn("- consultar_base_conhecimento_ti", mensagens)


class TestToolRag(unittest.TestCase):

    def test_resposta_preserva_fontes_e_pipeline_unico(self):
        # A tool DEVE chamar o pipeline da CP2 (buscar_com_fontes) — buscar_chunks
        # não é mais importado nem usado aqui (não existe em agent.tools).
        self.assertFalse(hasattr(tools_mod, "buscar_chunks"))
        with patch.object(
            tools_mod, "base_indexada", return_value=True
        ) as _, patch.object(
            tools_mod,
            "buscar_com_fontes",
            return_value=("Resposta RAG simulada.", [_doc_fonte("ping")]),
        ) as stub_rag:
            resultado = tools_mod.consultar_base_conhecimento_ti.invoke(
                {"consulta": "como diagnosticar a rede com ping?"}
            )

        # Fontes preservadas com documento/página no texto final
        self.assertIn("Fontes:", resultado)
        self.assertIn("01_cisco_troubleshooting_tcp_ip.pdf", resultado)
        self.assertIn("página 12", resultado)
        self.assertIn("Resposta RAG simulada", resultado)
        # Pipeline da CP2 (recuperação + reranking + geração) executado
        # APENAS uma vez por consulta
        self.assertEqual(stub_rag.call_count, 1)

    def test_base_nao_indexada_retorna_instrucao(self):
        with patch.object(tools_mod, "base_indexada", return_value=False):
            resultado = tools_mod.consultar_base_conhecimento_ti.invoke(
                {"consulta": "x"}
            )
        self.assertIn("python -m rag.main", resultado)
        self.assertIn("nomic-embed-text", resultado)

    def test_sem_informacao_na_base_nao_inventa(self):
        with patch.object(
            tools_mod, "base_indexada", return_value=True
        ), patch.object(
            tools_mod, "buscar_com_fontes", return_value=("", [])
        ):
            resultado = tools_mod.consultar_base_conhecimento_ti.invoke(
                {"consulta": "assunto fora da base"}
            )
        self.assertIn("não contém informações", resultado)

    def test_erro_de_recuperacao_degrada_com_instrucao(self):
        with patch.object(
            tools_mod, "base_indexada", return_value=True
        ), patch.object(
            tools_mod, "buscar_com_fontes", side_effect=RuntimeError("chroma off")
        ):
            resultado = tools_mod.consultar_base_conhecimento_ti.invoke(
                {"consulta": "x"}
            )
        self.assertIn("[ERRO]", resultado)
        self.assertIn("python -m rag.main", resultado)


class TestToolsEstruturadas(unittest.TestCase):

    def test_classificar_chamado_ti(self):
        llm_falso = FakeListChatModel(
            responses=[
                '{"categoria": "rede", "prioridade": "alta", '
                '"sistema_afetado": "Wi-Fi corporativo", '
                '"justificativa": "Trabalho parado e sem contorno.", '
                '"informacoes_faltantes": ["Há quanto tempo ocorre?"]}'
            ]
        )
        with patch.object(tools_mod, "build_llm_app", return_value=llm_falso):
            resultado = tools_mod.classificar_chamado_ti.invoke(
                {"descricao_problema": "sem internet"}
            )
        self.assertIn("Categoria: rede", resultado)
        self.assertIn("Prioridade sugerida: alta", resultado)
        self.assertIn("Há quanto tempo ocorre?", resultado)
        self.assertIn("NÃO foi registrado em nenhum sistema", resultado)

    def test_gerar_checklist_diagnostico(self):
        llm_falso = FakeListChatModel(
            responses=[
                "PARA ESCLARECER:\n- Em qual aplicativo?\n"
                "1. Reinicie o aplicativo."
            ]
        )
        with patch.object(tools_mod, "build_llm_app", return_value=llm_falso):
            resultado = tools_mod.gerar_checklist_diagnostico.invoke(
                {"descricao_problema": "app trava"}
            )
        self.assertIn("PARA ESCLARECER:", resultado)
        self.assertIn("1. Reinicie o aplicativo.", resultado)

    def test_verificar_informacoes_minimas(self):
        llm_falso = FakeListChatModel(
            responses=[
                '{"tem_sistema_ou_equipamento": false, '
                '"tem_sintoma_claro": true, '
                '"tem_inicio_da_ocorrencia": false, '
                '"tem_criticidade_relacionada": false, '
                '"informacoes_faltantes": ["Qual aplicativo?"], '
                '"resumo_validade": "Relato incompleto para triagem."}'
            ]
        )
        with patch.object(tools_mod, "build_llm_app", return_value=llm_falso):
            resultado = tools_mod.verificar_informacoes_minimas.invoke(
                {"descricao_problema": "não abre"}
            )
        self.assertIn("Presentes no relato:", resultado)
        self.assertIn("Qual aplicativo?", resultado)
        self.assertIn("Relato incompleto para triagem.", resultado)
        # Não afirma consulta a sistemas externos
        self.assertIn("nada foi", resultado.lower())


class TestAgentExecutorSelecao(unittest.TestCase):
    """End-to-end offline: create_react_agent + AgentExecutor com LLM fake,
    selecionando (a) a tool RAG e (b) nenhuma ferramenta."""

    @staticmethod
    def _executor(respostas: list[str]) -> AgentExecutor:
        agente = create_react_agent(
            llm=FakeListChatModel(responses=respostas),
            tools=FERRAMENTAS,
            prompt=REACT_PROMPT,
        )
        return AgentExecutor(
            agent=agente,
            tools=FERRAMENTAS,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=MAX_ITERACOES,
            return_intermediate_steps=True,
        )

    def test_seleciona_tool_rag(self):
        respostas = [
            "Thought: preciso consultar os documentos.\n"
            "Action: consultar_base_conhecimento_ti\n"
            "Action Input: como diagnosticar rede com ping\n",
            "Thought: agora sei a resposta.\n"
            "Final Answer: Use ping conforme a base. Fontes: Cisco TCP/IP.",
        ]
        executor = self._executor(respostas)
        with patch.object(
            tools_mod, "base_indexada", return_value=True
        ), patch.object(
            tools_mod,
            "buscar_com_fontes",
            return_value=("resp rag fake.", [_doc_fonte("ping")]),
        ):
            saida = executor.invoke(
                {"input": "como diagnosticar rede com ping?", "chat_history": []}
            )

        self.assertEqual(
            saida["output"],
            "Use ping conforme a base. Fontes: Cisco TCP/IP.",
        )
        self.assertEqual(
            saida["intermediate_steps"][0][0].tool,
            "consultar_base_conhecimento_ti",
        )

    def test_sem_ferramenta_para_conversa_geral(self):
        respostas = [
            "Thought: é um cumprimento, resposta direta.\n"
            "Final Answer: Olá! Em que posso ajudar com TI?",
        ]
        executor = self._executor(respostas)
        saida = executor.invoke({"input": "olá, tudo bem?", "chat_history": []})
        self.assertEqual(saida["output"], "Olá! Em que posso ajudar com TI?")
        self.assertEqual(len(saida["intermediate_steps"]), 0)

    def test_passos_internos_nao_contam_como_ferramenta(self):
        """
        Quando o parser ReAct falha, o AgentExecutor insere um passo
        interno "_Exception" no intermediate_steps (handle_parsing_errors).
        A ferramenta de resumo da interface NÃO deve contá-lo como uso de
        ferramenta — apenas ferramentas reais de FERRAMENTAS contam.
        """
        from agent.agent import AgenteTriagemTI

        agente_fake = FakeListChatModel(
            responses=[
                "sem formato ReAct nenhum aqui\n",  # força erro de parsing
                "Thought: agora sei a resposta.\n"
                "Final Answer: resposta direta sem ferramenta.",
            ]
        )
        executor = AgentExecutor(
            agent=create_react_agent(
                llm=agente_fake, tools=FERRAMENTAS, prompt=REACT_PROMPT
            ),
            tools=FERRAMENTAS,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=MAX_ITERACOES,
            return_intermediate_steps=True,
        )
        with patch(
            "agent.agent.construir_agente", return_value=executor
        ):
            stub = AgenteTriagemTI()
            resposta, ferramentas = stub.enviar_mensagem_com_passos("oi")
        self.assertEqual(resposta, "resposta direta sem ferramenta.")
        # Nenhuma ferramenta REAL foi usada (_Exception é interno e filtrado)
        self.assertEqual(ferramentas, [])



if __name__ == "__main__":
    unittest.main()
