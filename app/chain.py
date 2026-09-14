import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

from app.memory_manager import aplicar_memoria_do_projeto
from app.prompts import SYSTEM_PROMPT_CHAT, SYSTEM_PROMPT_EXTRACAO
from app.schemas import AnaliseChamado

load_dotenv()

MODELO = "gemma4:cloud"


def _get_api_key() -> str:
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OLLAMA_API_KEY não encontrada. Copie .env.example para .env "
            "e preencha sua chave da Ollama Cloud."
        )
    return api_key


def build_llm(temperature: float = 0.3) -> ChatOllama:
    """Instancia o ChatOllama apontando para o modelo gemma4:cloud."""
    base_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
    return ChatOllama(
        model=MODELO,
        base_url=base_url,
        temperature=temperature,
        client_kwargs={"headers": {"Authorization": f"Bearer {_get_api_key()}"}},
    )


class ConversationTI:


    def __init__(self) -> None:
        self.llm = build_llm()

        self.historico_completo: list[BaseMessage] = []

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT_CHAT),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )

        self._chain = prompt | self.llm

    def enviar_mensagem(self, mensagem_usuario: str) -> str:
        # Memória gerenciada: aplica o TokenBuffer sobre o histórico
        # completo, gerando a janela que efetivamente vai para o prompt.
        historico_gerenciado = aplicar_memoria_do_projeto(
            self.historico_completo, self.llm
        )

        resposta = self._chain.invoke(
            {"history": historico_gerenciado, "input": mensagem_usuario}
        )

        self.historico_completo.append(HumanMessage(content=mensagem_usuario))
        self.historico_completo.append(AIMessage(content=resposta.content))

        return resposta.content


def build_conversation_chain() -> ConversationTI:
    """Fábrica da Chain 1, usada pela interface (main.py)."""
    return ConversationTI()


def build_extraction_chain():
    """
    Chain 2 — Pipeline LCEL para saída estruturada.
    Usa o operador `|` explicitamente: prompt | llm | parser.
    """
    llm = build_llm(temperature=0.0)
    parser = PydanticOutputParser(pydantic_object=AnaliseChamado)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT_EXTRACAO + "\n\n{format_instructions}"),
            ("human", "Histórico da conversa de suporte:\n\n{conversa}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    # Arquitetura LCEL exigida: ChatPromptTemplate | ChatOllama | OutputParser
    extraction_chain = prompt | llm | parser
    return extraction_chain


def formatar_historico_para_extracao(mensagens: list[tuple[str, str]]) -> str:

    linhas = []
    for autor, texto in mensagens:
        prefixo = "Usuário" if autor == "user" else "Assistente"
        linhas.append(f"{prefixo}: {texto}")
    return "\n".join(linhas)