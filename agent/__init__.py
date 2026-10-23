"""
Pacote agent/ — CP3.

Agente inteligente ReAct (LangChain AgentExecutor + create_react_agent,
via langchain-classic, por causa do bug do langchain.agents no Python 3.14
— ver app/memory_manager.py) que reutiliza o pipeline RAG da CP2 (rag/)
e o chat de triagem da CP1 (app/) sem duplicar nenhuma etapa.

Submódulos:
    schemas  — modelos Pydantic das tools de triagem.
    prompts  — system prompt do agente + prompt ReAct + prompts das tools.
    tools    — ferramentas (@tool), incluindo a tool RAG da CP2.
    agent    — construção do AgentExecutor e gestão de histórico.
    main     — CLI de demonstração/teste do agente no terminal.
"""
