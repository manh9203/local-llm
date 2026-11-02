import pytest
from transformers import AutoTokenizer

from local_llm.chat_template import (
    fix_qwen3_chat_template,
    fix_qwen35_chat_template,
)


@pytest.mark.parametrize(
    "model_id,patch_fn",
    [
        (
            "mlx-community/Qwen3-8B-4bit",
            fix_qwen3_chat_template,
        ),
        (
            "mlx-community/Qwen3.5-9B-4bit",
            fix_qwen35_chat_template,
        ),
    ],
)
def test_non_thinking_prefix_consistency(model_id, patch_fn):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    patch_fn(tokenizer)

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
        history
        + [
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

    assert next_prompt[: len(first_prompt)] == first_prompt


def test_qwen35_thinking_mode_unchanged():
    tokenizer = AutoTokenizer.from_pretrained("mlx-community/Qwen3.5-9B-4bit")

    history = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is MLX?"},
        {
            "role": "assistant",
            "content": "MLX is a machine learning framework.",
        },
        {"role": "user", "content": "Who developed it?"},
    ]

    before = tokenizer.apply_chat_template(
        history,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_dict=False,
    )

    fix_qwen35_chat_template(tokenizer)

    after = tokenizer.apply_chat_template(
        history,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_dict=False,
    )

    assert before == after
