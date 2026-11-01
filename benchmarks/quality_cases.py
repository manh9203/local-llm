import hashlib
import json
from dataclasses import asdict, dataclass

Message = dict[str, str]

COMPACTION_INSTRUCTIONS = (
    "\nAnswer based only on the conversation. Return only a JSON "
    "object with exactly the fields and types/choices below. "
    "Use null or unknown for unspecified information when allowed. "
    "Do not include explanations.\n"
)


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
            answer_format={"context_tokens": "integer",
                           "generation_tokens": "integer"},
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
            answer_format={
                "transport": "train | plane | car | unknown", "budget_cad": "integer"},
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
            expected={"approved": "university_email",
                      "deferred": "government_id"},
            answer_format={
                "approved": "university_email | government_id | unknown",
                "deferred": "university_email | government_id | unknown",
            },
        ),
        Probe(
            name="next_priority",
            question="What is the immediate next priority?",
            expected={"priority": "verification_testing"},
            answer_format={
                "priority": "verification_testing | reviews | listing_filters | unknown"},
        ),
        Probe(
            name="mobile_scope",
            question="Is a mobile app in the approved project scope?",
            expected={"in_scope": False},
            answer_format={"in_scope": "boolean or null if unspecified"},
        ),
        Probe(
            name="completion_status",
            question=(
                "What completion status is explicitly confirmed for "
                "university-email verification and listing filters?"
            ),
            expected={"verification": "not_confirmed",
                      "listing_filters": "completed"},
            answer_format={
                "verification": "completed | not_confirmed",
                "listing_filters": "completed | not_confirmed",
            },
        ),
    ),
)


ALL_CASES = (TECHNICAL, TRAVEL, PROJECT)


@dataclass(frozen=True)
class CapabilityCase:
    id: str
    category: str
    messages: tuple[Message, ...]
    max_new_tokens: int
    grader: dict


CAPABILITY_CASES = (
    CapabilityCase(
        id="recall_corrected_config",
        category="recall",
        messages=tuple(message for segment in TECHNICAL.segments for message in segment) + ({
            "role": "user",
            "content": (
                "Give the current context limit, generation reservation, and checkpoint "
                "hash algorithm. Return only JSON with context_tokens (integer), "
                "generation_tokens (integer), and algorithm (sha256 | sha1 | unknown)."
            ),
        },),
        max_new_tokens=256,
        grader={"kind": "json", "expected": {
            "context_tokens": 8192, "generation_tokens": 512, "algorithm": "sha256",
        }},
    ),
    CapabilityCase(
        id="instructions_two_lines",
        category="instructions",
        messages=({"role": "user", "content": (
            "Describe removing an expired cache entry in exactly two nonempty lines. "
            "Start the first line with 'Action: ' and the second with 'Check: '. "
            "Use at most 32 words in total. Do not use the words 'fast' or 'easy'. "
            "Do not add bullets, headings, code fences, or a preamble."
        )},),
        max_new_tokens=256,
        grader={"kind": "constraints", "line_prefixes": ["Action: ", "Check: "],
                "max_words": 32, "forbidden_words": ["fast", "easy"]},
    ),
    CapabilityCase(
        id="python_lru_cache",
        category="python",
        messages=({"role": "user", "content": (
            "Implement a Python LRUCache class with __init__(capacity), get(key), and "
            "put(key, value). Reject nonpositive capacity with ValueError. get returns "
            "None for a missing key and makes an existing key most recently used. "
            "put inserts or updates a key, marks it most recently used, and evicts "
            "the least recently used key only when capacity is exceeded. Support "
            "hashable keys and arbitrary values, including None. Both operations "
            "must be O(1) on average. Standard-library collections are allowed. "
            "Return one complete implementation in a Python code block."
        )},),
        max_new_tokens=1024,
        grader={"kind": "manual", "rubric": [
            "Nonpositive capacity raises ValueError.",
            "Hits promote recency; misses return None without changing order.",
            "Updating a key promotes it without spuriously evicting another key.",
            "Insertion beyond capacity evicts the least recently used key.",
            "Stored None values work; get and put have average O(1) complexity.",
        ], "reference": (
            "At capacity 2: put(a,1), put(b,2), get(a), put(c,3) must evict b. "
            "Updating a must preserve c. Inspect both behavior and implementation. "
            "No code has been executed by this benchmark."
        )},
    ),
    CapabilityCase(
        id="cpp_move_only_buffer",
        category="cpp",
        messages=({"role": "user", "content": (
            "Write a complete C++17 Buffer class owning a dynamically allocated array "
            "of int. Its explicit size constructor must zero-initialize elements; "
            "zero length is valid. Delete copying. Implement noexcept move construction "
            "and move assignment, leaving the source empty with size() == 0. "
            "Self-move assignment must preserve the object. Provide size() and const "
            "and non-const operator[] for valid indices; bounds checking is not needed. "
            "Use std::unique_ptr<int[]> and include required headers. Return one C++ "
            "code block, with no additional text."
        )},),
        max_new_tokens=1024,
        grader={"kind": "manual", "rubric": [
            "Array ownership uses unique_ptr<int[]> and elements are zero-initialized.",
            "Zero-length construction and destruction are valid; copying is deleted.",
            "Move operations are noexcept and transfer ownership without leaks.",
            "Moved-from size is zero and self-move preserves the destination.",
            "Complete C++17 headers, size(), and both indexing overloads are present.",
        ], "reference": (
            "Inspect explicit move handling: moving the pointer alone does not reset "
            "the source's scalar size. Move assignment must release the old allocation. "
            "No compilation or generated-code execution has been performed."
        )},
    ),
    CapabilityCase(
        id="systems_memory_budget",
        category="systems",
        messages=({"role": "user", "content": (
            "Consider a hypothetical 16 GiB machine. Reserve 4 GiB for the OS, 6 GiB "
            "for model weights, and 1 GiB for other buffers. Divide all remaining "
            "memory equally among three cache sessions. Each cached token costs "
            "128 KiB. Based solely on memory, how many whole tokens fit per session? "
            "Do not impose a model context limit. Also state how many simultaneous "
            "generation requests are safe if per-request working memory and "
            "scheduling constraints have not been specified. Use binary units. "
            "Return only JSON with max_tokens_per_session (integer) and "
            "safe_inflight_requests (integer, or null if it cannot be determined)."
        )},),
        max_new_tokens=256,
        grader={"kind": "json", "expected": {
            "max_tokens_per_session": 13653, "safe_inflight_requests": None,
        }},
    ),
    CapabilityCase(
        id="recall_release_decisions",
        category="recall",
        messages=turn(
            "For the Aurora release, deploy in Toronto, retain logs for 30 days, "
            "and enable daily backups. We have not assigned an operations owner.",
            "Those are the release settings so far.",
        ) + turn(
            "Correction: deploy in Montreal and retain logs for 14 days. "
            "Disable backups for this release. A colleague suggested Maya as owner, "
            "but no owner has been approved.",
            "I will use the corrections and keep the proposed owner unassigned.",
        ) + ({"role": "user", "content": (
            "Return only the current approved release settings as JSON with "
            "region (montreal | toronto | unknown), retention_days (integer), backups_enabled (boolean), "
            "and owner (string or null when unassigned)."
        )},),
        max_new_tokens=256,
        grader={"kind": "json", "expected": {
            "region": "montreal", "retention_days": 14,
            "backups_enabled": False, "owner": None,
        }},
    ),
    CapabilityCase(
        id="instructions_redacted_json",
        category="instructions",
        messages=({"role": "user", "content": (
            'Sanitize this synthetic incident record: {"ticket":"INC-047", '
            '"internal_pin":"8492", "attempts":2, "limit":2, "resolved":false}. '
            "Return only a JSON object with exactly ticket, state, and retry. "
            "Copy ticket. Set state to closed if resolved is true, otherwise "
            "waiting if attempts has reached limit, otherwise ready. Set retry "
            "to true only when unresolved and attempts is below limit. "
            "Never include internal_pin, other fields, or surrounding commentary."
        )},),
        max_new_tokens=256,
        grader={"kind": "json", "expected": {
            "ticket": "INC-047", "state": "waiting", "retry": False,
        }},
    ),
    CapabilityCase(
        id="python_ttl_cache",
        category="python",
        messages=({"role": "user", "content": (
            "Implement a Python TTLCache class with __init__(ttl, clock), "
            "put(key, value), and get(key). ttl is a positive number of seconds; "
            "nonpositive ttl must raise ValueError. clock is an injected callable "
            "returning monotonic seconds; do not sleep or use the real clock. "
            "Each put sets that key's expiration to clock() + ttl, including "
            "updates. get returns None for a missing key or when clock() is "
            "greater than or equal to the key's expiration. Discard expired "
            "entries when accessed. A successful get must not extend expiration. "
            "Keys expire independently. Support hashable keys and arbitrary "
            "values, including None, with average O(1) get and put. Return one "
            "complete implementation in a Python code block, with no extra text."
        )},),
        max_new_tokens=1024,
        grader={"kind": "manual", "rubric": [
            "Nonpositive ttl raises ValueError; the injected clock is used.",
            "Expiration is checked at now >= deadline, including equality.",
            "put refreshes a key's expiration and get never extends it.",
            "Keys expire independently and expired entries are discarded on access.",
            "Hashable keys, arbitrary values, and stored None work with average O(1) operations.",
        ], "reference": (
            "With ttl 10, a value inserted at time 100 must be absent at time 110, "
            "even if read at time 105. Updating at 105 must extend its deadline "
            "to 115 without changing another key's expiration. Use a fake clock "
            "for all tests; inspect complexity separately."
        )},
    ),
    CapabilityCase(
        id="cpp_parse_int32",
        category="cpp",
        messages=({"role": "user", "content": (
            "Implement bool parse_int32(std::string_view text, std::int32_t& value) "
            "noexcept in C++17. Accept a nonempty sequence of ASCII decimal digits "
            "with one optional leading + or -; leading zeros are allowed. Reject "
            "empty input, a sign without digits, whitespace, nondecimal characters, "
            "trailing junk, and values outside the signed 32-bit range. On success "
            "assign value and return true. On every failure return false and leave "
            "value unchanged. Avoid arithmetic overflow and include all required "
            "headers. Standard-library parsing functions are allowed. Return one "
            "complete C++ code block with no main function or additional text."
        )},),
        max_new_tokens=1024,
        grader={"kind": "manual", "rubric": [
            "Accepts decimal digits, optional single signs, and leading zeros.",
            "Rejects empty/sign-only input, whitespace, junk, and nondecimal text.",
            "Accepts INT32_MIN and INT32_MAX and rejects values beyond both bounds.",
            "Preserves the caller's output on every failure without overflowing internally.",
            "Provides a complete C++17 noexcept implementation and required headers.",
        ], "reference": (
            "0, +0, -0, 0012, -2147483648, and 2147483647 are valid. "
            "2147483648, -2147483649, +-1, and 12x are invalid and must not "
            "modify the output. Check very long digit sequences without overflow."
        )},
    ),
    CapabilityCase(
        id="systems_context_budget",
        category="systems",
        messages=({"role": "user", "content": (
            "A request has an 8192-token context limit and reserves 768 tokens "
            "for generation. Its input contains 256 system tokens, an 896-token "
            "summary, 4100 tokens of recent turns, and an 1800-token incoming "
            "message. These counts already include all template overhead. "
            "Return only JSON with input_budget (integer), remaining_input_tokens "
            "(integer), and extra_turns (integer): the maximum whole additional "
            "completed turns that fit if each adds exactly 250 input tokens. "
            "Do not shorten the existing input or reduce the generation reservation."
        )},),
        max_new_tokens=256,
        grader={"kind": "json", "expected": {
            "input_budget": 7424, "remaining_input_tokens": 372, "extra_turns": 1,
        }},
    ),
)


def validate_grader(grader: dict) -> None:
    if not isinstance(grader, dict):
        raise TypeError("Grader must be an object")
    kind = grader.get("kind")
    if kind == "json":
        expected = grader.get("expected")
        if not isinstance(expected, dict) or not expected or any(
            not isinstance(key, str) or not key
            or type(value) not in (str, int, bool, type(None))
            for key, value in expected.items()
        ):
            raise ValueError(
                "Expected values must be a nonempty scalar object")
    elif kind == "constraints":
        for field in ("line_prefixes", "forbidden_words"):
            values = grader.get(field)
            if not isinstance(values, list) or not values or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError(f"{field} must contain nonempty strings")
        if type(grader.get("max_words")) is not int or grader["max_words"] < 1:
            raise ValueError("max_words must be a positive integer")
    elif kind == "manual":
        rubric = grader.get("rubric")
        if not isinstance(rubric, list) or not rubric or any(
            not isinstance(item, str) or not item.strip() for item in rubric
        ) or not isinstance(grader.get("reference"), str) or not grader["reference"].strip():
            raise ValueError("Manual grading requires a rubric and reference")
    else:
        raise ValueError(f"Unsupported grader: {kind}")


def validate_messages(messages, final_role: str) -> None:
    if not messages or any(not isinstance(message, dict) for message in messages):
        raise ValueError("Messages must contain message objects")
    offset = int(messages[0].get("role") == "system")
    for index, message in enumerate(messages):
        role = "system" if index < offset else (
            "user" if (index - offset) % 2 == 0 else "assistant"
        )
        if message.get("role") != role or not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"Invalid message at index {index}")
    if messages[-1]["role"] != final_role:
        raise ValueError(f"Last message must be {final_role}")


def select_cases(suite: str, case_id: str = "all", category: str = "all") -> tuple:
    cases = CAPABILITY_CASES if suite == "capability" else ALL_CASES
    if suite not in ("capability", "compaction"):
        raise ValueError(f"Unknown suite: {suite}")
    selected = tuple(case for case in cases if (
        case_id == "all" or (case.id if suite ==
                             "capability" else case.name) == case_id
    ) and (category == "all" or (
        case.category if suite == "capability" else "compaction"
    ) == category))
    if not selected:
        raise ValueError(
            f"No {suite} cases match case={case_id}, category={category}")
    return selected


def case_manifest(suite: str, cases: tuple, max_probes: int | None = None) -> list[dict]:
    if suite not in ("capability", "compaction"):
        raise ValueError(f"Unknown suite: {suite}")
    manifest = []
    ids = set()
    for case in cases:
        definition = asdict(case)
        case_id = case.id if suite == "capability" else case.name
        try:
            if not isinstance(case_id, str) or not case_id or case_id in ids:
                raise ValueError("Case IDs must be nonempty and unique")
            ids.add(case_id)
            if suite == "capability":
                if case.category not in ("recall", "instructions", "python", "cpp", "systems"):
                    raise ValueError(f"Unknown category: {case.category}")
                if type(case.max_new_tokens) is not int or not 0 < case.max_new_tokens < 8192:
                    raise ValueError(
                        "Generation budget must be between 1 and 8191")
                validate_messages(case.messages, "user")
                validate_grader(case.grader)
            else:
                definition["id"] = definition.pop("name")
                definition.update(category="compaction", max_new_tokens=128,
                                  instructions=COMPACTION_INSTRUCTIONS)
                if not case.segments:
                    raise ValueError("Compaction case needs segments")
                for segment in case.segments:
                    validate_messages(segment, "assistant")
                    if len(segment) < 4 or any(message["role"] == "system" for message in segment):
                        raise ValueError(
                            "Each segment needs at least two complete turns")
                probes = case.probes[:max_probes]
                if not probes or len({probe.name for probe in probes}) != len(probes):
                    raise ValueError("Probe names must be nonempty and unique")
                for probe in probes:
                    validate_grader(
                        {"kind": "json", "expected": probe.expected})
                    if not probe.name or not probe.question.strip() or probe.expected.keys() != probe.answer_format.keys() or any(
                        not isinstance(value, str) or not value.strip() for value in probe.answer_format.values()
                    ):
                        raise ValueError(
                            "Probe needs a name, question, and matching answer fields")
                definition["probes"] = definition["probes"][:max_probes]
            manifest.append(definition)
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"{case_id}: {error}") from error
    if not manifest:
        raise ValueError("Select at least one case")
    # Match the JSON representation saved in raw runs.
    return json.loads(json.dumps(manifest))


def case_set_hash(suite: str, definitions: list[dict]) -> str:
    data = json.dumps({"suite": suite, "cases": definitions}, sort_keys=True,
                      ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()
