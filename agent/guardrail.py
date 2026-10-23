"""
guardrail.py — Bônus de guardrail de entrada (CP3).

Valida a mensagem do usuário ANTES que o agente (ou qualquer ferramenta)
seja executado. Nenhum LLM, nenhum serviço externo: apenas heurísticas
determinísticas de palavras-chave/expressões regulares.

Ordem das regras:
  1. ENTRADA VAZIA — vazio ou só espaços  -> recusa.
  2. TAMANHO — acima de MAX_CARACTERES_ENTRADA -> recusa.
  3. PROMPT INJECTION — padrões conhecidos de "ignore instruções",
     extração de prompt de sistema / jailbreak -> recusa.
  4. FORA DO DOMÍNIO — marcadores claros de assunto não-TI presentes
     SEM nenhum sinal de domínio de TI -> recusa educada.

Princípio de abertura: se nenhuma regra de recusa casar de forma clara,
a entrada é APROVADA — evita bloquear perguntas legítimas de TI que a
heurística de palavras-chave não cobre. Perguntas legítimas de TI que
contenham, por acaso, um marcador off-topic continuam passando porque
precisam existir SINAIS de TI para a recusa.

Retorno: VereditoGuardrail (aprovado, motivo, mensagem_ao_usuario).
A mensagem_ao_usuario só é preenchida quando aprovado=False.
"""

import re
import unicodedata
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

MAX_CARACTERES_ENTRADA = 2000

ORIGEM = "TI.Assist"

_REJEITA_VAZIA = (
    "Recebi uma mensagem vazia. Escreva sua pergunta ou descreva o "
    "problema de TI que está enfrentando — por exemplo: \"meu notebook "
    "não conecta ao Wi-Fi corporativo\"."
)
_REJEITA_TAMANHO = (
    "Sua mensagem ficou muito longa (acima do limite desta triagem). "
    "Resuma o problema essencial — qual o sintoma, em qual sistema ou "
    "equipamento e desde quando — e eu assumo a partir daí."
)
_REJEITA_INJECAO = (
    "Essa mensagem parece uma tentativa de alterar minhas instruções "
    "internas, e eu só posso conversar dentro do meu papel de triagem "
    "de suporte de TI. Se você tem um problema técnico real (hardware, "
    "software, rede ou acesso), descreva o problema diretamente que eu "
    "te ajudo."
)
_REJEITA_FORA_DOMINIO = (
    "Essa conversa parece fora do meu escopo: sou assistente de triagem "
    "de suporte técnico de TI corporativo (hardware, software, rede e "
    "acesso a sistemas). Para assuntos de outros setores (RH, jurídico, "
    "financeiro, etc.), procure o canal responsável da empresa. Se o "
    "assunto envolver algum problema técnico, reformule com detalhes do "
    "que está acontecendo e eu assumo."
)

# ---------------------------------------------------------------------------
# Normalização de texto (sem acento, minúscula, qualquer espaço simples)
# ---------------------------------------------------------------------------

def _normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto)
    sem_acento = "".join(c for c in sem_acento if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", sem_acento.lower().strip())


# ---------------------------------------------------------------------------
# Regra 3 — padrões de prompt injection (PT e EN)
# ---------------------------------------------------------------------------

_PADROES_INJECAO = [
    # "ignore as instruções/anteriores/sistema..." (varições PT e EN)
    r"ignor[ae]\s+(as\s+|as\s+vossas\s+|todas\s+as\s+|todos\s+os\s+)?"
    r"(instru?(c|ç)(o|õ)es|regras|diretrizes|sistema|restrictions|guidelines)",
    r"ignor[ae]\s+(all\s+|any\s+|your\s+|the\s+)?"
    r"(previous\s+|prior\s+|above\s+)?(instructions?|rules|prompts?)",
    r"desconsidere\s+(as\s+|todas\s+as\s+)?(instru?(c|ç)(o|õ)es|regras)",
    r"esque(c|ç)a\s+(as\s+|todas\s+as\s+)?(instru?(c|ç)(o|õ)es|regras)",
    r"disregard\s+(all\s+|your\s+|the\s+)?(previous\s+|prior\s+)?instructions?",
    # extração do prompt de sistema (3 palavras quaisquer entre o verbo e
    # o alvo cobrem "mostre me suas ...", "revele o seu ..." etc.)
    r"(revele?|mostre|imprima|repita|echo)\s+(?:[a-z]+\s+){0,3}"
    r"(prompt\s+de\s+sistema|system\s+prompt|prompt\s+inicial|"
    r"instruc?oes\s+do\s+sistema|instru?cao?es\s+do\s+sistema)",
    r"system\s+prompt",
    # tentativas de fuga de papel conhecidas
    r"jailbreak|\bdan\s+mode\b|dev(?:elo)?per\s+mode\b|modo\s+desenvolvedor",
]

_INJECAO_RE = [re.compile(p, re.IGNORECASE) for p in _PADROES_INJECAO]


# ---------------------------------------------------------------------------
# Regra 4 — domínio: sinais de TI x marcadores off-topic
# ---------------------------------------------------------------------------

_SINAIS_DOMINIO_TI = [
    "notebook", "computador", " pc ", "laptop", "desktop", "tablet",
    "wifi", "wi-fi", "internet", "rede", "roteador", "modem", "cabo",
    "impressora", "scanner", "teclado", "mouse", "monitor", "tela",
    "headset", "fone de ouvido", "microfone", "webcam", "bateria",
    "carregador", "fonte de alimenta", "cracha",
    "senha", "login", "logoff", "acesso", "vpn",
    "2fa", "mfa", "token", "autentica", "certificado",
    "windows", "linux", "macos", "android", "ios",
    "driver", "atualiza", "software", "aplicativo", "aplicacao",
    "programa", "sistema", "excel", "word",
    "powerpoint", "outlook", "teams", "sharepoint", "email", "e-mail",
    "chrome", "edge", "firefox", "navegador", "site", "portal",
    "antivirus", "firewall", "backup", "banco de dados",
    "servidor", "pdf", "arquivo", "pasta", "disco",
    "ssd", "hd ", "memoria", "ram", "cpu", "usb",
    "som", "audio", "mudo", "bluetooth", "java",
    "trava", "travando", "lento", "nao liga", "nao abre",
    "reiniciar", "formatar", "permiss",
    "helpdesk", "chamado", "triagem", "suporte", " ti ", " de ti",
]

_MARCADORES_FORA_DOMINIO = [
    # gastronomia
    "receita", "bolo", "massa de pizza", "pizza", "hamburguer",
    "almoco", "jantar", "sobremesa", "brigadeiro",
    # esporte/entretenimento puro
    "futebol", "campeonato", "partida de", "time de futebol", "gol do",
    "jogo do", "palpite", "loteria", "bingo",
    # ocultismo/lifestyle
    "horoscopo", "signo", "tarot", "videncia",
    # outros departamentos / vida pessoal
    "folha de pagamento", "salario", "ferias", "folgas",
    "licenca maternidade", "beneficios", "contracheque",
    "advogad", "juridico", "processo judicial", "audiencia",
    "planejamento financeiro", "investimento", "cripto", "bitcoin",
    "viagem", "hotel", "passagem de aviao", "passaporte",
    # curiosidades genéricas de estudo fora de TI
    "matematica", "equacao de", "historia do brasil",
    "curiosidade sobre animais", "receita de sabao",
]

_SINAIS_TI_RE = re.compile(
    "|".join(_SINAIS_DOMINIO_TI), re.IGNORECASE
)
_MARCADORES_RE = re.compile(
    "|".join(_MARCADORES_FORA_DOMINIO), re.IGNORECASE
)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VereditoGuardrail:
    """Resultado da validação de entrada (antes de tocar no agente)."""

    aprovado: bool
    motivo: str                      # "ok" | "vazia" | "tamanho" | "injecao" | "fora_dominio"
    mensagem_ao_usuario: str = ""    # preenchida só quando aprovado=False


def validar_entrada(texto: str) -> VereditoGuardrail:
    """Valida a mensagem do usuário ANTES de qualquer ferramenta rodar."""
    if texto is None or not texto.strip():
        return VereditoGuardrail(False, "vazia", _REJEITA_VAZIA)

    if len(texto.strip()) > MAX_CARACTERES_ENTRADA:
        return VereditoGuardrail(False, "tamanho", _REJEITA_TAMANHO)

    normalizado = _normalizar(texto)

    if normalizado == "":
        return VereditoGuardrail(False, "vazia", _REJEITA_VAZIA)

    for padrao in _INJECAO_RE:
        if padrao.search(normalizado):
            return VereditoGuardrail(False, "injecao", _REJEITA_INJECAO)

    tem_sinal_de_TI = _SINAIS_TI_RE.search(normalizado) is not None
    tem_marcador_off = _MARCADORES_RE.search(normalizado) is not None

    if tem_marcador_off and not tem_sinal_de_TI:
        return VereditoGuardrail(False, "fora_dominio", _REJEITA_FORA_DOMINIO)

    # Default: aprovar (evita bloquear perg legítimas de TI).
    return VereditoGuardrail(True, "ok", "")
