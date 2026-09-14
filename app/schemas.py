from enum import Enum

from pydantic import BaseModel, Field, field_validator


class CategoriaChamado(str, Enum):


    HARDWARE = "hardware"
    SOFTWARE = "software"
    REDE = "rede"
    ACESSO = "acesso"
    OUTRO = "outro"


class UrgenciaChamado(str, Enum):


    BAIXA = "baixa"
    MEDIA = "media"
    ALTA = "alta"
    CRITICA = "critica"


class AnaliseChamado(BaseModel):


    categoria: CategoriaChamado = Field(
        description="Categoria técnica do problema relatado pelo usuário."
    )
    urgencia: UrgenciaChamado = Field(
        description=(
            "Nível de urgência do chamado, considerando impacto no trabalho "
            "do usuário e quantidade de pessoas afetadas."
        )
    )
    sistema_afetado: str = Field(
        description="Sistema, software, equipamento ou serviço afetado.",
        min_length=2,
    )
    resumo_problema: str = Field(
        description="Resumo objetivo do problema relatado, em até 2 frases.",
        min_length=10,
    )
    acao_recomendada: str = Field(
        description=(
            "Próxima ação recomendada: uma orientação de autoatendimento "
            "ou o motivo para encaminhar a um técnico humano."
        )
    )
    requer_escalonamento: bool = Field(
        description=(
            "True se o chamado precisa ser escalado para um técnico humano; "
            "False se pode ser resolvido apenas com as orientações dadas."
        )
    )

    @field_validator("resumo_problema", "acao_recomendada")
    @classmethod
    def campos_texto_nao_vazios(cls, valor: str) -> str:
        
        valor_limpo = valor.strip()
        if not valor_limpo:
            raise ValueError("Campo de texto não pode ser vazio.")
        return valor_limpo