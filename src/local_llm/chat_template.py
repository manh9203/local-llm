def fix_qwen3_chat_template(tokenizer):
    """Keep non-thinking assistant history consistent with generation."""
    template = tokenizer.chat_template

    if not isinstance(template, str):
        raise TypeError("Expected a single Qwen3 chat template.")

    old = """        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}"""

    new = """        {%- elif enable_thinking is defined and enable_thinking is false %}
            {{- '<|im_start|>' + message.role + '\\n<think>\\n\\n</think>\\n\\n' + content }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}"""

    if template.count(old) != 1:
        raise ValueError(
            "Unexpected Qwen3 template structure. "
            "Refusing to modify the template."
        )

    tokenizer.chat_template = template.replace(old, new, 1)


def fix_qwen35_chat_template(tokenizer):
    """Preserve non-thinking prefixes in historical assistant turns."""
    original = tokenizer.chat_template

    old = """        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- endif %}"""

    new = """        {%- elif enable_thinking is defined and enable_thinking is false %}
            {{- '<|im_start|>' + message.role + '\\n<think>\\n\\n</think>\\n\\n' + content }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- endif %}"""

    if original.count(old) != 1:
        raise ValueError(
            "Unexpected Qwen3.5 chat template; "
            "cannot safely apply prefix-preservation patch"
        )

    tokenizer.chat_template = original.replace(old, new, 1)
