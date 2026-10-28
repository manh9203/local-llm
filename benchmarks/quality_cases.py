from dataclasses import dataclass

Message = dict[str, str]


def turn(user: str, assistant: str) -> tuple[Message, ...]:
    return (
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    )


@dataclass(frozen=True)
class Probe:
    name: str
    question: str
    expected: dict[str, str | int | bool | None]
    # Describes valid outputs without identifying the correct selections.
    answer_format: dict[str, str]


@dataclass(frozen=True)
class QualityCase:
    name: str
    segments: tuple[tuple[Message, ...], ...]
    probes: tuple[Probe, ...]


TECHNICAL = QualityCase(
    name="technical",
    segments=(
        turn(
            "I'm building a local LLM assistant on an Apple M1 "
            "with 16 GB unified memory. The model is "
            "Qwen3-8B-4bit through MLX-LM. My context limit "
            "is currently 4096 tokens.",
            "We can build around those hardware constraints.",
        ) + turn(
            "Unrelated detail: I made lemon pasta yesterday.",
            "Sounds good. We can return to the inference project.",
        ),
        turn(
            "Correction: the current context limit is now "
            "8192 tokens, with 512 tokens reserved for generation. "
            "The input budget is 7680 tokens. We decided to use "
            "SHA-256 fingerprints of message prefixes for "
            "compaction checkpoints.",
            "I'll use the updated configuration and checkpoint design.",
        ) + turn(
            "I also watched a documentary last weekend.",
            "Understood.",
        ),
        turn(
            "We retain up to 8192 KV-cache tokens and preserve "
            "four recent completed turns during compaction. "
            "The current priority is evaluating summary quality "
            "before comparing Qwen3.5 models.",
            "We'll evaluate quality first.",
        ) + turn(
            "I reorganized my desktop icons today.",
            "Okay.",
        ),
    ),
    probes=(
        Probe(
            name="current_context",
            question=(
                "What are the current context limit and "
                "reserved generation budget?"
            ),
            expected={"context_tokens": 8192, "generation_tokens": 512},
            answer_format={"context_tokens": "integer", "generation_tokens": "integer"},
        ),
        Probe(
            name="checkpoint_design",
            question="How are compaction checkpoints identified?",
            expected={"algorithm": "sha256", "input": "message_prefix"},
            answer_format={
                "algorithm": "sha256 | sha1 | md5 | unknown",
                "input": "message_prefix | conversation_id | unknown",
            },
        ),
        Probe(
            name="recent_turns",
            question=(
                "How many recent completed turns "
                "are preserved during compaction?"
            ),
            expected={"completed_turns": 4},
            answer_format={"completed_turns": "integer"},
        ),
    ),
)


TRAVEL = QualityCase(
    name="travel",
    segments=(
        turn(
            "I'm planning a weekend trip to Kingston. "
            "I prefer taking the train rather than flying. "
            "My tentative budget is CAD 400.",
            "We can plan around rail travel and that budget.",
        ) + turn(
            "I recently bought a blue notebook for class.",
            "Nice.",
        ),
        turn(
            "Change of plans: the destination is Ottawa, "
            "not Kingston. The budget is now CAD 500. "
            "We'll leave Friday evening and return Sunday. "
            "We still prefer the train.",
            "I'll use Ottawa and the updated schedule and budget.",
        ) + turn(
            "I saw a funny movie trailer this morning.",
            "Sounds entertaining.",
        ),
        turn(
            "We need accommodation near downtown. "
            "A friend suggested a museum, but we haven't "
            "booked any museum tickets yet.",
            "I'll treat the museum as a suggestion, not a booking.",
        ) + turn(
            "I should probably replace my backpack soon.",
            "That can be handled separately.",
        ),
    ),
    probes=(
        Probe(
            name="destination",
            question="What is the current trip destination?",
            expected={"destination": "ottawa"},
            answer_format={"destination": "city name or unknown"},
        ),
        Probe(
            name="transport_budget",
            question="What transport do we prefer and what's the current budget?",
            expected={"transport": "train", "budget_cad": 500},
            answer_format={"transport": "train | plane | car | unknown", "budget_cad": "integer"},
        ),
        Probe(
            name="museum_booking",
            question="Have we booked museum tickets?",
            expected={"booked": False},
            answer_format={"booked": "boolean or null if unspecified"},
        ),
        Probe(
            name="museum_decision",
            question="Have we decided against booking museum tickets, or is that undecided?",
            expected={"decision": "undecided"},
            answer_format={"decision": "book | do_not_book | undecided"},
        ),
        Probe(
            name="hotel_name",
            question="What is the name of our booked hotel?",
            expected={"hotel_name": None},
            answer_format={"hotel_name": "string or null if unspecified"},
        ),
    ),
)


PROJECT = QualityCase(
    name="project",
    segments=(
        turn(
            "Our team is building a web-based student housing prototype. "
            "The demo was originally scheduled for Monday. "
            "We need user verification and listing filters.",
            "Those features can support the initial demo.",
        ) + turn(
            "Separately, I've been listening to jazz lately.",
            "Nice.",
        ),
        turn(
            "The demo has moved to Wednesday. We decided "
            "to implement university-email verification now. "
            "Government-ID verification is deferred. "
            "A mobile app was suggested but isn't in our scope.",
            "I'll treat the email approach as approved and "
            "the ID approach as deferred.",
        ) + turn(
            "I changed the wallpaper on my laptop.",
            "Okay.",
        ),
        turn(
            "Listing filters are complete. Reviews are still "
            "blocked on policy decisions. The next priority "
            "is testing the verification flow before Wednesday.",
            "I'll keep verification testing as the immediate next step.",
        ) + turn(
            "I might buy a new desk lamp someday.",
            "Understood.",
        ),
    ),
    probes=(
        Probe(
            name="demo_deadline",
            question="What day is the demo currently scheduled for?",
            expected={"day": "wednesday"},
            answer_format={"day": "weekday name or unknown"},
        ),
        Probe(
            name="verification",
            question=(
                "Which verification method was approved, "
                "and which was deferred?"
            ),
            expected={"approved": "university_email", "deferred": "government_id"},
            answer_format={
                "approved": "university_email | government_id | unknown",
                "deferred": "university_email | government_id | unknown",
            },
        ),
        Probe(
            name="next_priority",
            question="What is the immediate next priority?",
            expected={"priority": "verification_testing"},
            answer_format={"priority": "verification_testing | reviews | listing_filters | unknown"},
        ),
        Probe(
            name="mobile_scope",
            question="Is a mobile app in the approved project scope?",
            expected={"in_scope": False},
            answer_format={"in_scope": "boolean or null if unspecified"},
        ),
    ),
)


ALL_CASES = (TECHNICAL, TRAVEL, PROJECT)
