# ai_assert

**Runtime constraint verification for AI outputs. 278 lines. Zero dependencies.**

```python
from ai_assert import ai_assert, valid_json, max_length, contains

result = ai_assert(
    prompt="Return a JSON object with a 'greeting' key",
    constraints=[valid_json(), max_length(200), contains("hello")],
    generate_fn=my_llm,    # any function: str → str
    max_retries=3,
)

print(result.output)       # guaranteed valid JSON, ≤200 chars, contains "hello"
print(result.passed)       # True
print(result.attempts)     # 1–4 (retried with feedback until constraints pass)
```

## The Problem

LLMs don't reliably follow instructions. You ask for JSON, you get markdown. You ask for 100 words, you get 500. You ask for a list of 5 items, you get 7.

Every AI application handles this with ad-hoc validation scattered across the codebase. Or worse — hopes for the best.

## The Solution

`ai_assert` is a universal **check → score → retry** loop:

1. **Generate**: Call your LLM (any model, any provider)
2. **Check**: Run every constraint (multiplicative gate — ALL must pass)
3. **Retry**: If any constraint fails, feed back failure details and regenerate
4. **Return**: Verified output with full audit trail

```
Prompt → LLM → Check all constraints → All pass? → Return ✓
                      ↓ (any fail)
              Build feedback prompt → Retry (up to max_retries)
```

## Key Properties

- **Zero dependencies** — stdlib only (`dataclasses`, `typing`, `json`, `functools`)
- **Model-agnostic** — works with OpenAI, Anthropic, local models, anything with `str → str`
- **278 lines** — small enough to read in one sitting, audit in an hour
- **Multiplicative gate** — a zero in ANY constraint = failure (not averaged away)
- **Continuous scoring** — each check returns a score in `[0, 1)`, never 1.0
- **Full audit trail** — every attempt, every check result, preserved in `result.history`

## Install

```bash
pip install ai-assert
```

Or just copy `ai_assert.py` into your project. It's one file with zero dependencies.

## Quick Start

### 1. Define your generator

Any function that takes a `str` and returns a `str`:

```python
from openai import OpenAI

client = OpenAI()

def my_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
```

### 2. Declare constraints

```python
from ai_assert import valid_json, max_length, min_length, contains, not_contains, matches_schema, custom

constraints = [
    valid_json(),                                          # must be parseable JSON
    max_length(500),                                       # at most 500 chars
    matches_schema({"required": ["name", "age"]}),         # must have these keys
    not_contains("```"),                                   # no markdown code fences
    custom("positive_age", lambda s: json.loads(s).get("age", 0) > 0,
           fail_msg="Age must be positive"),               # any boolean function
]
```

### 3. Call ai_assert

```python
from ai_assert import ai_assert

result = ai_assert(
    prompt="Generate a user profile as JSON with name and age.",
    constraints=constraints,
    generate_fn=my_llm,
    max_retries=3,
)

if result.passed:
    user = json.loads(result.output)  # safe — guaranteed valid
    print(f"Got {user['name']}, age {user['age']}")
else:
    print(f"Failed after {result.attempts} attempts")
    for check in result.checks:
        if not check.passed:
            print(f"  ✗ {check.name}: {check.message}")
```

### 4. Or use the decorator

```python
from ai_assert import reliable, valid_json, max_length

@reliable(constraints=[valid_json(), max_length(500)], max_retries=3)
def generate_profile(prompt: str) -> str:
    return my_llm(prompt)

result = generate_profile("Generate a user profile as JSON.")
# result is an AiAssertResult — same interface as ai_assert()
```

## Built-in Constraints

| Constraint | Description |
|---|---|
| `max_length(n)` | Output ≤ n characters |
| `min_length(n)` | Output ≥ n characters |
| `contains(s)` | Output contains substring s |
| `not_contains(s)` | Output does NOT contain substring s |
| `valid_json()` | Output is parseable JSON |
| `matches_schema(s)` | Output is JSON with required keys |
| `custom(name, fn)` | Any `str → bool` function |

### Custom constraints

```python
from ai_assert import Constraint

def word_count_between(min_w, max_w):
    def check(output):
        count = len(output.split())
        if min_w <= count <= max_w:
            return True, 0.99, f"{count} words"
        return False, 0.0, f"{count} words, need {min_w}-{max_w}"
    return Constraint(name=f"word_count({min_w}-{max_w})", check_fn=check)
```

## The AiAssertResult

```python
@dataclass
class AiAssertResult:
    output: str                  # the final output
    passed: bool                 # did all constraints pass?
    checks: list[CheckResult]    # per-constraint results
    attempts: int                # how many tries it took
    history: list[dict]          # full audit trail of all attempts

    @property
    def composite_score(self) -> float:   # average score, always < 1.0
```

Each `CheckResult`:
```python
@dataclass
class CheckResult:
    name: str       # constraint name
    passed: bool    # pass/fail
    score: float    # continuous in [0, 1), never 1.0
    message: str    # human-readable explanation
```

## Benchmark Results

Tested on [IFEval](https://arxiv.org/abs/2311.07911) (541 prompts, 25 constraint types):

| Metric | Baseline (single-pass) | With ai_assert | Improvement |
|---|---|---|---|
| Prompt-level accuracy | 69.3% | 76.2% | **+6.8pp** |
| Constraint-level accuracy | 77.5% | 82.5% | **+5.0pp** |

- 30 prompts rescued from failure by the retry mechanism
- All improvements from runtime verification — no model changes

## Design Decisions

**Why multiplicative, not averaged?** If your JSON is invalid, it doesn't matter that the length was perfect. A zero in any dimension is system failure. This is the multiplicative gate — `all(c.passed for c in checks)`.

**Why scores in [0, 1) and not [0, 1]?** No verification achieves perfect fidelity. The score participates in correctness without claiming identity with it. This prevents false confidence in downstream aggregation.

**Why retry with feedback, not just Best-of-N?** Feedback-directed retry is more sample-efficient. The failure messages tell the model exactly what to fix. Best-of-N generates blindly N times and picks the best. (Both approaches compose — you can use ai_assert inside a Best-of-N loop.)

**Why zero dependencies?** So you can drop it into any project. No framework lock-in. No transitive dependency hell. Copy the file and go.

## Works With

ai_assert works with any `str → str` function. Tested with:

- **OpenAI** (GPT-4o, GPT-4o-mini)
- **Anthropic** (Claude 3.5 Sonnet)
- **Local models** (Ollama, vLLM, llama.cpp)
- **Any HTTP API** (wrap the call in a function)

See [examples/basic_usage.py](examples/basic_usage.py) for integration code.

## Compared To

| Feature | ai_assert | Guardrails AI | Instructor | OpenAI Structured Outputs |
|---|---|---|---|---|
| Zero dependencies | ✅ | ❌ | ❌ | ❌ |
| Semantic constraints | ✅ | ❌ | ❌ | ❌ |
| Model-agnostic | ✅ | ✅ | ❌ (OpenAI) | ❌ (OpenAI) |
| Retry with feedback | ✅ | ✅ | ✅ | ❌ |
| Continuous scoring | ✅ | ❌ | ❌ | ❌ |
| Lines of code | 278 | ~15,000+ | ~5,000+ | N/A (server-side) |

## Contributing

Issues and PRs welcome. The codebase is 278 lines — reading the entire thing takes less time than reading this README.

## License

MIT — see [LICENSE](LICENSE).
