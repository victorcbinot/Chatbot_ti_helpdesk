"""
test_compressao_cp3.py — Testes OFFLINE do bônus de compressão de
contexto do SCRATCHPAD do agente ReAct (agent/scratchpad_compressor.py).

O que é provado aqui (e o que NÃO):
  - A compressão acontece DE DENTRO do AgentExecutor: quando as observações
    acumuladas (Action, Observation) ultrapassam o limite de caracteres,
    as antigas viram UMA observação sintética (síntese extrativa) e só a
    janela de recentes fica intacta — o loop continua e termina com Final
    Answer (agente continua funcionando).
  - Fontes ("Fonte/Fontes" + documento + página) e conteúdo técnico são
    preservados VERBATIM na síntese (sem re-escrever via LLM, sem alucinar
    citação).
  - Nível de integração: `construir_agente()` (caminho de produção) retorna
    um AgentExecutorComCompressao.
  - DISTINÇÃO: nada aqui é "limitar o histórico visual do Gradio" — a
    compressão é do scratchpad interno {agent_scratchpad} de um ÚNICO
    turno (o histórico de conversa entre turnos é outro mecanismo,
    aplicar_token_buffer da CP1).

Execução:   python -m unittest discover -s tests -v   (na raiz do projeto)
Requisitos: venv com requirements.txt instalado. NÃO exige Ollama
            local/cloud nem OLLAMA_API_KEY (tudo simulado).
"""

import os
import unittest
from unittest.mock import patch

from langchain_classic.agents import create_react_agent
from langchain_core.agents import AgentAction
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import tool

import agent.scratchpad_compressor as comp_mod
from agent.agent import MAX_ITERACOES, construir_agente
from agent.prompts import REACT_PROMPT
from agent.scratchpad_compressor import (
    AgentExecutorComCompressao,
    comprimir_intermediate_steps,
    limite_caracteres,
)


def _acao(nome: str, consulta: str = "consulta") -> AgentAction:
    return AgentAction(
        tool=nome,
        tool_input=consulta,
        log=f"Thought: preciso do {nome}.\nAction: {nome}\nAction Input: {consulta}",
    )


def _passo(nome: str, observacao: str, consulta: str = "consulta") -> tuple:
    return (_acao(nome, consulta), observacao)


def _observacao_longa(tamanho: int = 1500, fonte: str | None = None) -> str:
    corpo = "Diagnóstico detalhado: " + "conteúdo técnico relevante. " * (tamanho // 30)
    if fonte:
        corpo += f"\nFontes:\n{fonte}"
    return corpo


FONTE_A = "1. 05_cisa_phishing_resistant_mfa.pdf (página 4)"
FONTE_B = "2. 01_cisco_troubleshooting_tcp_ip.pdf (página 12)"


class TestCompressaoUnitaria(unittest.TestCase):
    """Comportamento da função comprimir_intermediate_steps."""

    def test_abaixo_do_limite_e_identidade_exacta(self):
        passos = [
            _passo("consultar_base_conhecimento_ti", "observação curta."),
            _passo("classificar_chamado_ti", "integridade x acesso."),
        ]
        saida = comprimir_intermediate_steps(passos)
        # MESMOS objetos (sem recriação) — nada foi comprimido
        self.assertTrue(all(a is b for a, b in zip(passos, saida)))

    def test_mais_passos_que_limite_pequeno_mas_dentro_do_total(self):
        passos = [_passo("gerar_checklist_diagnostico", "passo %d ok." % i) for i in range(6)]
        saida = comprimir_intermediate_steps(passos)
        self.assertEqual(len(saida), 6)
        self.assertTrue(all(a is b for a, b in zip(passos, saida)))

    def test_compressao_acima_do_limite_mantem_janela_recente_intacta(self):
        nomes = ["consultar_base_conhecimento_ti", "classificar_chamado_ti",
                 "gerar_checklist_diagnostico", "verificar_informacoes_minimas",
                 "consultar_base_conhecimento_ti"]
        passos = [_passo(nome, _observacao_longa(1500), f"consulta {nome}") for nome in nomes]

        saida = comprimir_intermediate_steps(passos)
        janela = comp_mod.janela_recentes()

        self.assertEqual(len(saida), janela + 1)
        sintetico, obs_sintese = saida[0]
        self.assertEqual(sintetico.tool, "resumo_de_observacoes")
        # Janela recente: MESMOS objetos, intacta e no fim
        for saida_passo, original_passo in zip(saida[1:], passos[-janela:]):
            self.assertIs(saida_passo, original_passo)
        # A síntese menciona as ferramentas antigas (não apagadas as buscas)
        self.assertIn("consultar_base_conhecimento_ti", obs_sintese)
        self.assertIn("classificar_chamado_ti", obs_sintese)

    def test_sintese_contem_marcador_e_preserva_fontes_verbatim(self):
        # duas observações LONGAS com blocos de fontes + 2 pequenas:
        # total >> limite (3000), com a janela recente (2) ficando intacta
        # e o resumo contendo os DOIS blocos de fontes verbatim.
        passos = [
            _passo("consultar_base_conhecimento_ti",
                   _observacao_longa(2500, fonte=FONTE_A)),
            _passo("consultar_base_conhecimento_ti",
                   _observacao_longa(2500, fonte=FONTE_B)),
            _passo("classificar_chamado_ti", "categoria: Rede"),
            _passo("gerar_checklist_diagnostico", "verificar VPN"),
        ]
        saida = comprimir_intermediate_steps(passos)
        sintese = saida[0][1].replace("\n", " | ")
        self.assertIn("[SÍNTESE COMPACTADA", sintese)
        # Fontes VERBATIM preservadas dentro da síntese
        self.assertIn(FONTE_A, sintese)
        self.assertIn(FONTE_B, sintese)

    def test_sintese_nao_ultrapassa_o_teto(self):
        teto = limite_caracteres()
        passos = [
            _passo("consultar_base_conhecimento_ti", "padrão técnico " * 4000)
            for _ in range(5)
        ]
        saida = comprimir_intermediate_steps(passos)
        self.assertLessEqual(len(saida[0][1]), teto + 40)

    def test_limite_e_janela_via_variaveis_de_ambiente(self):
        passos = [
            _passo("classificar_chamado_ti", "observação %d ." % i + "texto" * 80)
            for i in range(4)
        ]

        with patch.dict(os.environ, {"SCRATCHPAD_LIMITE_CARACTERES": "800",
                                     "SCRATCHPAD_JANELA_RECENTES": "1"}):
            saida = comprimir_intermediate_steps(passos)

        self.assertEqual(len(saida), 2)  # 1 síntese + 1 recente
        self.assertIs(saida[-1], passos[-1])
        self.assertEqual(saida[0][0].tool, "resumo_de_observacoes")
        # O teto é garantido para a PARTE sintética; a janela recente fica
        # intacta de propósito (pode exceder o limite — é a informação fresca).
        self.assertLessEqual(len(saida[0][1]), 800 + 40)


class TestExecutorIntegrado(unittest.TestCase):
    """Compressão engatada DE DENTRO do loop real do AgentExecutor."""

    @staticmethod
    @tool
    def ferramenta_longa(consulta: str) -> str:
        """Ferramenta de teste que devolve uma observação longa com fontes."""
        return _observacao_longa(1600, fonte=FONTE_A)

    def _executor(self, respostas):
        llm = FakeListChatModel(responses=respostas)
        agente = create_react_agent(
            llm=llm, tools=[self.ferramenta_longa], prompt=REACT_PROMPT,
        )
        return AgentExecutorComCompressao(
            agent=agente,
            tools=[self.ferramenta_longa],
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=12,
            return_intermediate_steps=True,
        )

    def _respostas_de_cinco_passos(self):
        acao = (
            "Thought: preciso consultar os documentos.\n"
            "Action: ferramenta_longa\n"
            "Action Input: consulta genérica de rede\n"
        )
        return [acao] * 5 + [
            "Thought: com as observações na íntegra e a síntese, tenho.\n"
            "Final Answer: Diagnóstico concluído com base nas cinco buscas.",
        ]

    def test_compressao_engatada_no_loop_e_agente_termina_funcionando(self):
        chamadas: list[dict] = []
        original = comp_mod.comprimir_intermediate_steps

        def intercecao(passos, *args, **kwargs):
            saida = original(passos, *args, **kwargs)
            # CÓPIA na entrada: a lista do executor continua crescente depois.
            chamadas.append({
                "total_entrada": comp_mod._tamanho_passos(passos),
                "entrada": list(passos),
                "saida": saida,
            })
            return saida

        # Intercepta com o MESMO comportamento real; apenas registra.
        with patch.object(comp_mod, "comprimir_intermediate_steps", side_effect=intercecao):
            executor = self._executor(self._respostas_de_cinco_passos())
            saida = executor.invoke({"input": "diagnosticar rede", "chat_history": []})

        self.assertEqual(
            saida["output"], "Diagnóstico concluído com base nas cinco buscas."
        )
        # Compressão REALMENTE necessária: há iteração em que o scratchpad,
        # acumulado pelo executor, passou do limite de caracteres.
        self.assertGreaterEqual(len(chamadas), 6)
        self.assertTrue(any(c["total_entrada"] > limite_caracteres() for c in chamadas))
        # Em ao menos UMA iteração a saída contém a síntese (acao sintética)
        saidas_sinteticas = [
            c for c in chamadas
            if c["saida"] and c["saida"][0][0].tool == "resumo_de_observacoes"
        ]
        self.assertGreaterEqual(len(saidas_sinteticas), 1)
        chamada = saidas_sinteticas[0]
        janela = comp_mod.janela_recentes()
        self.assertIn("[SÍNTESE COMPACTADA", chamada["saida"][0][1])
        # Estrutura: 1 observação sintética + janela de recentes intacta
        self.assertEqual(len(chamada["saida"]), 1 + janela)
        # A janela recente são os MESMOS objetos do scratchpad acumulado
        for saida_passo, entrada_passo in zip(chamada["saida"][1:], chamada["entrada"][-janela:]):
            self.assertIs(saida_passo, entrada_passo)

    def test_sem_compressao_quando_passos_ausentes(self):  # scratchpad vazio
        executor = self._executor(
            ["Thought: direta.\nFinal Answer: resposta imediata."]
        )
        saida = executor.invoke({"input": "olá", "chat_history": []})
        self.assertEqual(saida["output"], "resposta imediata.")
        # Nenhum passo sintético apareceu (nada a comprimir)
        for acao_passo, _ in saida["intermediate_steps"]:
            self.assertNotEqual(acao_passo.tool, "resumo_de_observacoes")


class TestIntegracaoProducao(unittest.TestCase):
    """O caminho de produção (construir_agente) usa o executor comprimido."""

    def test_construir_agente_retorna_executor_com_compressao(self):
        with patch("agent.agent.build_llm_app",
                   return_value=FakeListChatModel(
                       responses=["Thought: direta.\nFinal Answer: ok."])):
            executor = construir_agente()
        self.assertIsInstance(executor, AgentExecutorComCompressao)

        saida = executor.invoke({"input": "oi", "chat_history": []})
        self.assertEqual(saida["output"], "ok.")
        self.assertLessEqual(MAX_ITERACOES, executor.max_iterations)


class TestTetoFalhasFormato(unittest.TestCase):
    """
    Teto de falhas de formato ReAct consecutivas (respostas vazias/malformadas
    do modelo): em vez de queimar todas as iterações até o cru
    "Agent stopped due to iteration limit or time limit.", o executor
    encerra o turno com mensagem amigável — e ainda tolera até
    MAX_FALHAS_FORMATO_CONSECUTIVAS - 1 falhas seguidas para o modelo
    se recuperar no mesma turno (comportamento observado em produção).
    """

    @staticmethod
    @tool
    def ferramenta_curta(consulta: str) -> str:
        """Ferramenta trivial de teste."""
        return "observação curta."

    def _executor(self, respostas):
        llm = FakeListChatModel(responses=respostas)
        agente = create_react_agent(
            llm=llm, tools=[self.ferramenta_curta], prompt=REACT_PROMPT,
        )
        return AgentExecutorComCompressao(
            agent=agente,
            tools=[self.ferramenta_curta],
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=12,
            return_intermediate_steps=True,
        )

    def test_muitas_falhas_consecutivas_terminam_com_mensagem_amigavel(self):
        saida = self._executor(["resposta sem formato ReAct nenhum."]).invoke(
            {"input": "pergunta", "chat_history": []}
        )
        self.assertEqual(saida["output"], comp_mod.MSG_FALHA_FORMATO)
        # NÃO é o cru de stop do langchain nem loop até burnout (12)
        self.assertNotIn("Agent stopped due to", saida["output"])
        etapas = saida.get("intermediate_steps", [])
        self.assertLess(len(etapas), 6)

    def test_modelo_se_recupera_apos_uma_ou_duas_falhas(self):
        respostas = [
            "resposta sem formato ReAct nenhum.",
            "Thought: agora em formato válido.\nFinal Answer: recuperado.",
        ]
        saida = self._executor(respostas).invoke(
            {"input": "pergunta", "chat_history": []}
        )
        self.assertEqual(saida["output"], "recuperado.")

    def test_contador_de_falhas_consecutivas(self):
        _acao, obs = _passo("consultar_base_conhecimento_ti", "ok")
        etapas = [
            (AgentAction(tool="_Exception", tool_input=f"erro {i}", log="x"), f"obs {i}")
            for i in range(3)
        ] + [_passo("consultar_base_conhecimento_ti", "ok")]
        # Ferramenta real no fim: nenhuma falha consecutiva em andamento
        self.assertEqual(comp_mod.contar_falhas_formato_consecutivas(etapas), 0)
        # Uma exceção isolada no fim: 1
        self.assertEqual(
            comp_mod.contar_falhas_formato_consecutivas(
                [(AgentAction(tool="_Exception", tool_input="e", log="x"), "obs")]
            ),
            1,
        )
        # Três exceções seguidas no fim: 3 (atinge MAX_FALHAS_FORMATO_CONSECUTIVAS)
        self.assertEqual(
            comp_mod.contar_falhas_formato_consecutivas(etapas[:-1]), 3
        )


if __name__ == "__main__":
    unittest.main()
