
from transformers import AutoTokenizer

from local_llm.chat_template import fix_qwen3_chat_template


def test_non_thinking_prefix_consistency():
    tokenizer = AutoTokenizer.from_pretrained(
        "mlx-community/Qwen3-8B-4bit"
    )
    fix_qwen3_chat_template(tokenizer)

    history = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is MLX?"},
    ]

    first_prompt = tokenizer.apply_chat_template(
        history,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )

    next_prompt = tokenizer.apply_chat_template(
        history + [
            {
                "role": "assistant",
                "content": "MLX is a machine learning framework.",
            },
            {"role": "user", "content": "Who developed it?"},
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )

    assert next_prompt[:len(first_prompt)] == first_prompt
