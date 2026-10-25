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
    reference: str

    # Each group represents one required fact.
    # Alternative regex patterns within a group are OR'd.
    required: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class DownstreamCase:
    name: str
    segments: tuple[tuple[Message, ...], ...]
    probes: tuple[Probe, ...]


TECHNICAL = DownstreamCase(
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
            reference="8192 context tokens; 512 generation tokens.",
            required=((r"\b8,?192\b",), (r"\b512\b",)),
        ),
        Probe(
            name="checkpoint_design",
            question="How are compaction checkpoints identified?",
            reference="SHA-256 fingerprints of message prefixes.",
            required=((r"sha[\s-]?256",),),
        ),
        Probe(
            name="recent_turns",
            question=(
                "How many recent completed turns "
                "are preserved during compaction?"
            ),
            reference="Four completed turns.",
            required=((r"\b4\b", r"\bfour\b"),),
        ),
    ),
)


TRAVEL = DownstreamCase(
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
            reference="Ottawa.",
            required=((r"\bottawa\b",),),
        ),
        Probe(
            name="transport_budget",
            question="What transport do we prefer and what's the current budget?",
            reference="Train; CAD 500.",
            required=(
                (r"\btrain\b", r"\brail\b"),
                (r"\b500\b",),
            ),
        ),
        Probe(
            name="museum_booking",
            question="Have we booked museum tickets?",
            reference="No, museum tickets have not been booked.",
            required=((
                r"\bnot\s+(?:yet\s+)?booked\b",
                r"\bhaven't booked\b",
                r"\bno\s+(?:museum\s+)?tickets?\s+(?:have\s+been\s+)?booked\b",
                r"\bnot\s+confirmed\b",
            ),),
        ),
    ),
)


PROJECT = DownstreamCase(
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
            reference="Wednesday.",
            required=((r"\bwednesday\b",),),
        ),
        Probe(
            name="verification",
            question=(
                "Which verification method was approved, "
                "and which was deferred?"
            ),
            reference=(
                "University-email verification approved; "
                "government-ID verification deferred."
            ),
            required=(
                (r"\b(?:university|student)\s*[- ]?email\b",),
                (r"\bgovernment\s*[- ]?id\b",),
                (r"\bdefer", r"\bpostpon", r"\blater\b"),
            ),
        ),
        Probe(
            name="next_priority",
            question="What is the immediate next priority?",
            reference="Testing the verification flow.",
            required=((
                r"\btest(?:ing)?\b.{0,70}\bverif",
                r"\bverif\w*\b.{0,70}\btest(?:ing)?\b",
            ),),
        ),
    ),
)


ALL_CASES = (TECHNICAL, TRAVEL, PROJECT)
