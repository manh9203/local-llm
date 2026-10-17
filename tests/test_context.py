from local_llm.chat import ChatConfig
from local_llm.inference.engine import ChatSession


def test_context_truncation_removes_oldest_exchange():
    session = ChatSession.__new__(ChatSession)

    session.config = ChatConfig(
        max_context_tokens=60,
        max_new_tokens=20,
    )

    session.messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Old question"},
        {"role": "assistant", "content": "Old answer"},
        {"role": "user", "content": "Recent question"},
        {"role": "assistant", "content": "Recent answer"},
    ]

    # Fake token counts to test the history-removal policy.
    def fake_tokenize(messages):
        return list(range(len(messages) * 10))

    session.tokenize_messages = fake_tokenize

    pending, token_ids = session.prepare_prompt(
        "Latest question"
    )

    assert len(token_ids) <= 40

    assert pending == [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Recent question"},
        {"role": "assistant", "content": "Recent answer"},
        {"role": "user", "content": "Latest question"},
    ]

    # Preparing a prompt must not mutate stored history.
    assert session.messages[1]["content"] == "Old question"
