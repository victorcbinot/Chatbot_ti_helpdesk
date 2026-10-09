"""
main.py — Entry point do projeto CKP02 (python -m rag.main).

Interface de linha de comando: faz perguntas ao pipeline RAG sobre a base
de conhecimento real de TI, com a fonte citada na resposta.

Na primeira execução (ou se a base ainda não tiver sido indexada), monta
automaticamente os embeddings e o ChromaDB a partir dos PDFs em docs/,
para as 2 configurações de chunking.

Comandos especiais durante a conversa:
  fontes    -> lista os documentos da base de conhecimento
  config    -> troca qual configuração de chunking está sendo usada
  filtro    -> aplica/remove um filtro de metadata (categoria, fornecedor,
               tipo_documento) na busca, sem precisar informá-lo a cada pergunta
  rerank    -> ativa/desativa o reranking (cross-encoder) em runtime
  sair      -> encerra o programa
"""

from rag.embeddings_store import carregar_vectorstore, criar_vectorstore
from rag.loader import carregar_documentos, listar_documentos_fonte
from rag.metadata import (
    CAMPOS_FILTRAVEIS,
    construir_filtro,
    descrever_filtro,
    valores_disponiveis,
)
from rag.rag_chain import build_rag_chain
from rag.reranker import reranker_ativo
from rag.splitter import CONFIGURACOES_CHUNKING, dividir_com_todas_configuracoes

CONFIG_PADRAO = "pequeno_512"


def _base_ja_indexada(config_nome: str) -> bool:
    """Verifica se já existe uma coleção do ChromaDB com dados para essa configuração."""
    try:
        vectorstore = carregar_vectorstore(config_nome)
        dados = vectorstore.get(limit=1)
        return len(dados.get("ids", [])) > 0
    except Exception:
        return False


def construir_base_de_conhecimento() -> None:
    """Roda o pipeline load -> split -> embed -> store para as 2 configurações de chunking."""
    print("Construindo a base de conhecimento (isso só acontece uma vez)...\n")
    documentos = carregar_documentos()
    print()
    configuracoes = dividir_com_todas_configuracoes(documentos)

    print("\nGerando embeddings e armazenando no ChromaDB...")
    for nome, chunks in configuracoes.items():
        criar_vectorstore(chunks, nome)

    print("\nBase de conhecimento pronta.\n")


def garantir_base_pronta() -> None:
    """Garante que ao menos a configuração padrão esteja indexada antes de iniciar a conversa."""
    if not _base_ja_indexada(CONFIG_PADRAO):
        construir_base_de_conhecimento()


def main() -> None:
    print("=" * 60)
    print("DocMind RAG — Assistente de TI com base de conhecimento real")
    print("=" * 60)

    garantir_base_pronta()

    config_atual = CONFIG_PADRAO
    filtro_atual = None
    usar_rerank = None
    chain = build_rag_chain(config_atual, filtro=filtro_atual, rerank=usar_rerank)

    print(f"\nConfiguração de chunking ativa: {config_atual}")
    print(f"Filtro de metadata ativo: {descrever_filtro(filtro_atual)}")
    print(f"Reranking (cross-encoder): {'ATIVO' if reranker_ativo() else 'desativado'}")
    print("Comandos especiais: 'fontes', 'config', 'filtro', 'rerank', 'sair'\n")

    while True:
        pergunta = input("Você: ").strip()

        if not pergunta:
            continue

        if pergunta.lower() in ("sair", "exit", "quit"):
            print("\nDocMind: Até mais!")
            break

        if pergunta.lower() == "fontes":
            fontes = listar_documentos_fonte()
            print("\nDocumentos na base de conhecimento:")
            for f in fontes:
                print(f"  - {f}")
            print()
            continue

        if pergunta.lower() == "config":
            opcoes = list(CONFIGURACOES_CHUNKING.keys())
            print(f"\nConfigurações disponíveis: {opcoes}")
            nova = input(f"Qual usar? (atual: {config_atual}) ").strip()
            if nova in CONFIGURACOES_CHUNKING:
                config_atual = nova
                chain = build_rag_chain(config_atual, filtro=filtro_atual, rerank=usar_rerank)
                print(f"Configuração trocada para: {config_atual}\n")
            else:
                print("Configuração inválida, mantendo a atual.\n")
            continue

        if pergunta.lower() == "filtro":
            print("\nCampos filtráveis e valores disponíveis:")
            for campo in CAMPOS_FILTRAVEIS:
                print(f"  {campo}: {valores_disponiveis(campo)}")
            print(f"Filtro atual: {descrever_filtro(filtro_atual)}")

            entrada = input(
                "Filtro (ex: categoria=Rede, fornecedor=CISA) ou 'limpar': "
            ).strip()

            if entrada.lower() in ("limpar", "nenhum", ""):
                filtro_atual = None
                print("Filtro removido. Busca sem filtro.\n")
            else:
                try:
                    campos = {}
                    for parte in entrada.split(","):
                        chave, _, valor = parte.partition("=")
                        campos[chave.strip()] = valor.strip()
                    filtro_atual = construir_filtro(**campos)
                    print(f"Filtro aplicado: {descrever_filtro(filtro_atual)}\n")
                except ValueError as exc:
                    print(f"Filtro inválido: {exc}\n")

            chain = build_rag_chain(config_atual, filtro=filtro_atual, rerank=usar_rerank)
            continue

        if pergunta.lower() == "rerank":
            estado = "ATIVO" if reranker_ativo() else "desativado"
            print(f"\nReranking atual (env RERANKER_ATIVO): {estado}")
            escolha = input("Ativar, desativar ou voltar ao padrão? (on/off/padrao): ").strip().lower()
            if escolha in ("on", "ativar", "1", "true"):
                usar_rerank = True
                print("Reranking ativado.\n")
            elif escolha in ("off", "desativar", "0", "false"):
                usar_rerank = False
                print("Reranking desativado. Busca top_k direto no ChromaDB.\n")
            else:
                usar_rerank = None
                print(f"Voltou ao padrão do env (RERANKER_ATIVO={'true' if reranker_ativo() else 'false'}).\n")
            chain = build_rag_chain(config_atual, filtro=filtro_atual, rerank=usar_rerank)
            continue

        try:
            resposta = chain.invoke(pergunta)
            print(f"\nDocMind: {resposta}\n")
        except Exception as exc:  # noqa: BLE001 — mostramos o erro ao usuário
            print(f"\nErro ao gerar resposta: {exc}\n")


if __name__ == "__main__":
    main()