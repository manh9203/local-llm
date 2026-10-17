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
