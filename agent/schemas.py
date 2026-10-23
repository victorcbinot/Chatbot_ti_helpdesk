"""
schemas.py — Modelos Pydantic das tools de triagem da CP3.

Reaproveita os enums CategoriaChamado e UrgenciaChamado da CP1
(app/schemas.py) — mesma taxonomia de categorias, agora usada como
prioridade sugerida (baixa/média/alta/crítica) pela ferramenta de
classificação.
"""

from app.schemas import CategoriaChamado, UrgenciaChamado
from pydantic import BaseModel, Field, field_validator


class ClassificacaoChamado(BaseModel):
    """
    Saída estruturada da ferramenta classificar_chamado_ti:
    categoria, prioridade sugerida, justificativa e o que ainda precisa
    ser perguntado ao usuário.
    """

    categoria: CategoriaChamado = Field(
        description=(
            "Categoria técnica do problema relatado: hardware, software, "
            "rede, acesso ou outro."
        )
    )
    prioridade: UrgenciaChamado = Field(
        description=(
            "Prioridade sugerida para o chamado (baixa, media, alta, "
            "critica), definida pelos critérios de impacto e urgência."
        )
    )
    sistema_afetado: str = Field(
        description=(
            "Sistema, equipamento ou serviço afetado, extraído do relato. "
            "Se o relato não informar, use 'não informado' — nunca invente."
        )
    )
    justificativa: str = Field(
        description=(
            "Justificativa curta (1-2 frases) que conecta o relato à "
            "categoria e à prioridade escolhidas, citando o critério "
            "de impacto/urgência observado."
        ),
        min_length=10,
    )
    informacoes_faltantes: list[str] = Field(
        description=(
            "Lista do que ainda precisa ser perguntado ao usuário antes "
            "de fechar a triagem (ex.: sistema afetado, há quanto tempo "
            "ocorre, mensagem de erro). Vazia quando o relato está completo."
        ),
        default_factory=list,
    )

    @field_validator("justificativa", "sistema_afetado")
    @classmethod
    def campos_texto_nao_vazios(cls, valor: str) -> str:
        valor_limpo = valor.strip()
        if not valor_limpo:
            raise ValueError("Campo de texto não pode ser vazio.")
        return valor_limpo


class VerificacaoInformacoes(BaseModel):
    """
    Saída estruturada da ferramenta verificar_informacoes_minimas:
    quais informações mínimas o relato contém e o que falta.
    """

    tem_sistema_ou_equipamento: bool = Field(
        description=(
            "True se o relato identifica o sistema, aplicativo ou "
            "equipamento afetado."
        )
    )
    tem_sintoma_claro: bool = Field(
        description=(
            "True se o relato descreve o sintoma com clareza (o que "
            "exatamente não funciona ou está errado)."
        )
    )
    tem_inicio_da_ocorrencia: bool = Field(
        description=(
            "True se o relato indica quando o problema começou ou há "
            "quanto tempo ocorre."
        )
    )
    tem_criticidade_relacionada: bool = Field(
        description=(
            "True se o relato indica impacto no trabalho ou quantas "
            "pessoas são afetadas."
        )
    )
    informacoes_faltantes: list[str] = Field(
        description=(
            "Lista das informações ausentes, redigida como perguntas "
            "para o usuário (ex.: 'Em qual aplicativo o erro aparece?')."
        ),
        default_factory=list,
    )
    resumo_validade: str = Field(
        description=(
            "Resumo curto dizendo se o relato é suficiente para triagem "
            "ou o que falta — baseado EXCLUSIVAMENTE no texto fornecido, "
            "sem consultar nenhum sistema externo."
        ),
        min_length=10,
    )

    @field_validator("resumo_validade")
    @classmethod
    def resumo_nao_vazio(cls, valor: str) -> str:
        valor_limpo = valor.strip()
        if not valor_limpo:
            raise ValueError("Campo de texto não pode ser vazio.")
        return valor_limpo
