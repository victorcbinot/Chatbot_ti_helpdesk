"""
metadata.py — Diferencial de Metadata Filtering (CP2).

Define os metadados reais da base de conhecimento (categoria, fornecedor e
tipo de documento), associando cada PDF de docs/ aos seus valores, e oferece
helpers para montar filtros no formato ``where`` do ChromaDB.

Os valores abaixo são fundamentados no conteúdo real de cada documento
(ver a tabela do README.md) — nenhum metadado é inventado ou inferido
automaticamente a partir do texto.
"""

# Campos pelos quais a busca pode ser filtrada (viram chaves de metadata).
CAMPOS_FILTRAVEIS = ("categoria", "fornecedor", "tipo_documento")

# Categoria usada quando um PDF novo ainda não foi registrado aqui.
CATEGORIA_PADRAO = "Não classificado"

# Metadados por nome de arquivo em docs/, com base no conteúdo real dos PDFs.
METADADOS_POR_ARQUIVO: dict[str, dict[str, str]] = {
    "01_cisco_troubleshooting_tcp_ip.pdf": {
        "categoria": "Rede",
        "fornecedor": "Cisco",
        "tipo_documento": "Troubleshooting Guide",
    },
    "02_lenovo_hardware_maintenance_manual.pdf": {
        "categoria": "Hardware",
        "fornecedor": "Lenovo",
        "tipo_documento": "Hardware Maintenance Manual",
    },
    "03_oracle_java_troubleshooting_guide.pdf": {
        "categoria": "Software",
        "fornecedor": "Oracle",
        "tipo_documento": "Troubleshooting Guide",
    },
    "04_cisa_remote_access_software.pdf": {
        "categoria": "Acesso",
        "fornecedor": "CISA",
        "tipo_documento": "Guide",
    },
    "05_cisa_phishing_resistant_mfa.pdf": {
        "categoria": "Acesso",
        "fornecedor": "CISA",
        "tipo_documento": "Fact Sheet",
    },
}


def metadados_do_arquivo(nome_arquivo: str) -> dict[str, str]:
    """
    Retorna os metadados de um PDF de docs/.

    Se o arquivo não estiver registrado, devolve apenas a categoria padrão,
    para não quebrar o pipeline ao adicionar documentos novos.
    """
    if nome_arquivo in METADADOS_POR_ARQUIVO:
        return dict(METADADOS_POR_ARQUIVO[nome_arquivo])
    return {"categoria": CATEGORIA_PADRAO}


def valores_disponiveis(campo: str) -> list[str]:
    """Lista, em ordem, os valores distintos de um campo entre os documentos registrados."""
    if campo not in CAMPOS_FILTRAVEIS:
        raise ValueError(
            f"Campo '{campo}' não é filtrável. Use um de: {CAMPOS_FILTRAVEIS}."
        )
    return sorted({meta[campo] for meta in METADADOS_POR_ARQUIVO.values() if campo in meta})


def construir_filtro(**campos: str | None) -> dict | None:
    """
    Monta um filtro no formato ``where`` do ChromaDB a partir de campos
    opcionais. Retorna ``None`` quando nenhum campo é informado (busca sem
    filtro, comportamento original).

    Exemplos::

        construir_filtro(categoria="Rede")
        # -> {"categoria": "Rede"}

        construir_filtro(categoria="Acesso", fornecedor="CISA")
        # -> {"$and": [{"categoria": "Acesso"}, {"fornecedor": "CISA"}]}
    """
    desconhecidos = [c for c in campos if c not in CAMPOS_FILTRAVEIS]
    if desconhecidos:
        raise ValueError(
            f"Campo(s) não filtrável(is): {desconhecidos}. "
            f"Use um de: {CAMPOS_FILTRAVEIS}."
        )

    condicoes = [{campo: valor} for campo, valor in campos.items() if valor]
    if not condicoes:
        return None
    if len(condicoes) == 1:
        return condicoes[0]
    return {"$and": condicoes}


def descrever_filtro(filtro: dict | None) -> str:
    """Devolve uma descrição legível de um filtro ``where`` (ou 'sem filtro')."""
    if not filtro:
        return "sem filtro"

    condicoes = filtro.get("$and", [filtro])
    partes = [f"{campo}={valor}" for condicao in condicoes for campo, valor in condicao.items()]
    return ", ".join(partes)
