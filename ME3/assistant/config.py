"""Assistant configuration, read from ME3/.env (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"


def _b(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


def _opt_int(name: str):
    v = os.environ.get(name, "").strip()
    return int(v) if v else None


@dataclass
class Config:
    bot_name: str = "Iris"
    llm_provider: str = "anthropic"              # anthropic | openrouter
    llm_model: str = "claude-haiku-5-5"
    anthropic_api_key: str = ""
    openrouter_api_key: str = ""
    openrouter_sort: str = "latency"

    asr_mcp_url: str = "http://127.0.0.1:8101/mcp"
    tts_mcp_url: str = "http://127.0.0.1:8102/mcp"
    tool_mcp_urls: dict = field(default_factory=dict)   # name -> url (assist, home)
    agent_mcp_url: str = "http://127.0.0.1:8105/mcp"
    mcp_autostart: bool = True

    audio_in_sample_rate: int = 16000
    audio_out_sample_rate: int = 24000
    input_device_index: int | None = None
    output_device_index: int | None = None

    vad_confidence: float = 0.7
    vad_start_secs: float = 0.2
    vad_stop_secs: float = 0.2
    turn_stop: str = "smart"                      # smart (Smart Turn v3 model) | timeout
    speech_timeout: float = 0.6
    interrupt_min_words: int = 0                  # 0: any detected speech interrupts; N: need N words (anti-echo)

    enable_tools: bool = True
    enable_agent: bool = True
    greet: bool = True
    fillers: bool = True
    echo_filter: bool = True
    aec: bool = True
    aec_delay_ms: int = 60
    log_level: str = "INFO"
    turns_log: Path = LOGS / "turns.jsonl"
    latency_log: Path = LOGS / "latency.jsonl"

    def validate(self) -> list[str]:
        p = []
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            p.append("ANTHROPIC_API_KEY is empty (LLM_PROVIDER=anthropic)")
        if self.llm_provider == "openrouter" and not self.openrouter_api_key:
            p.append("OPENROUTER_API_KEY is empty (LLM_PROVIDER=openrouter)")
        if self.llm_provider not in {"anthropic", "openrouter"}:
            p.append(f"LLM_PROVIDER must be anthropic or openrouter, not {self.llm_provider!r}")
        if self.audio_in_sample_rate != 16000:
            p.append("AUDIO_IN_SAMPLE_RATE must be 16000")
        return p


def load() -> Config:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    except Exception:  # noqa: BLE001
        pass
    e = os.environ.get
    LOGS.mkdir(parents=True, exist_ok=True)
    tools = {"assist": e("ASSIST_MCP_URL", "http://127.0.0.1:8103/mcp"),
             "home": e("HOME_MCP_URL", "http://127.0.0.1:8104/mcp")}
    provider = e("LLM_PROVIDER", "openrouter").lower()
    return Config(
        bot_name=e("BOT_NAME", "Iris"),
        llm_provider=provider,
        llm_model=e("LLM_MODEL", "claude-haiku-5-5" if provider == "anthropic" else "openai/gpt-oss-120b"),
        anthropic_api_key=e("ANTHROPIC_API_KEY", ""),
        openrouter_api_key=e("OPENROUTER_API_KEY", ""),
        openrouter_sort=e("OPENROUTER_SORT", "latency"),
        asr_mcp_url=e("ASR_MCP_URL", "http://127.0.0.1:8101/mcp"),
        tts_mcp_url=e("TTS_MCP_URL", "http://127.0.0.1:8102/mcp"),
        tool_mcp_urls={k: v for k, v in tools.items() if v},
        agent_mcp_url=e("AGENT_MCP_URL", "http://127.0.0.1:8105/mcp"),
        mcp_autostart=_b("MCP_AUTOSTART", True),
        audio_in_sample_rate=int(e("AUDIO_IN_SAMPLE_RATE", 16000)),
        audio_out_sample_rate=int(e("AUDIO_OUT_SAMPLE_RATE", 24000)),
        input_device_index=_opt_int("INPUT_DEVICE_INDEX"),
        output_device_index=_opt_int("OUTPUT_DEVICE_INDEX"),
        vad_confidence=float(e("VAD_CONFIDENCE", 0.7)),
        vad_start_secs=float(e("VAD_START_SECS", 0.2)),
        vad_stop_secs=float(e("VAD_STOP_SECS", 0.2)),
        turn_stop=e("TURN_STOP", "smart").lower(),
        speech_timeout=float(e("SPEECH_TIMEOUT", 0.6)),
        interrupt_min_words=int(e("INTERRUPT_MIN_WORDS", 0)),
        enable_tools=_b("ENABLE_TOOLS", True),
        enable_agent=_b("ENABLE_AGENT", True),
        greet=_b("GREET", True),
        fillers=_b("FILLERS", True),
        echo_filter=_b("ECHO_FILTER", True),
        aec=_b("AEC", True),
        aec_delay_ms=int(e("AEC_DELAY_MS", 60)),
        log_level=e("LOG_LEVEL", "INFO"),
    )
