from dataclasses import dataclass

DEFAULT_MODEL_ID = "mlx-community/Qwen3.5-9B-4bit"


@dataclass(frozen=True)
class ChatConfig:
    model_id: str = DEFAULT_MODEL_ID
    system_prompt: str = "You are a helpful, concise AI assistant."
    max_context_tokens: int = 8192
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    enable_thinking: bool = False
    use_cache: bool = True
