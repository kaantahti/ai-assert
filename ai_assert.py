"""
ai_assert — Runtime AI output validation primitive.

A lightweight, universal constraint-checking layer for AI outputs.
Architecture: Context (Distinction) → Constraints (Selection) → Correction (Actualization)
"""

from dataclasses import dataclass, field
from typing import Callable, Any, Optional
import json
import functools


# ---------------------------------------------------------------------------
# Core Data Structures
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Result of a single constraint check.

    score is continuous in [0, 1) — never 1.0 (structural bound: T6).
    """
    name: str
    passed: bool
    score: float  # 0.0 = total failure, approaches but never reaches 1.0
    message: str = ""

    def __post_init__(self):
        # Enforce convergence bound: score must be in [0, 1)
        if self.score >= 1.0:
            self.score = 0.9999
        if self.score < 0.0:
            self.score = 0.0


@dataclass
class Constraint:
    """A named constraint with a check function.

    check_fn: takes output string, returns (passed: bool, score: float, message: str)
    """
    name: str
    check_fn: Callable[[str], tuple[bool, float, str]]

    def check(self, output: str) -> CheckResult:
        passed, score, message = self.check_fn(output)
        return CheckResult(
            name=self.name,
            passed=passed,
            score=score,
            message=message,
        )


@dataclass
class AiAssertResult:
    """Complete result of an ai_assert call, including retry history."""
    output: str
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)
    attempts: int = 1
    history: list[dict] = field(default_factory=list)

    @property
    def composite_score(self) -> float:
        """Weighted average of all check scores. Always < 1.0."""
        if not self.checks:
            return 0.0
        total = sum(c.score for c in self.checks) / len(self.checks)
        return min(total, 0.9999)  # Convergence bound


# ---------------------------------------------------------------------------
# Built-in Constraint Factories
# ---------------------------------------------------------------------------

def max_length(n: int) -> Constraint:
    """Output must be at most n characters."""
    def check(output: str) -> tuple[bool, float, str]:
        length = len(output)
        if length <= n:
            score = max(0.0, min(1.0 - (length / max(n * 2, 1)), 0.9999))
            return True, score, f"Length {length} <= {n}"
        return False, 0.0, f"Length {length} > {n}"
    return Constraint(name=f"max_length({n})", check_fn=check)


def min_length(n: int) -> Constraint:
    """Output must be at least n characters."""
    def check(output: str) -> tuple[bool, float, str]:
        length = len(output)
        if length >= n:
            score = min(length / max(n * 2, 1), 0.9999)
            return True, score, f"Length {length} >= {n}"
        return False, 0.0, f"Length {length} < {n}"
    return Constraint(name=f"min_length({n})", check_fn=check)


def contains(substring: str) -> Constraint:
    """Output must contain the given substring."""
    def check(output: str) -> tuple[bool, float, str]:
        if substring in output:
            return True, 0.99, f"Contains '{substring}'"
        return False, 0.0, f"Missing '{substring}'"
    return Constraint(name=f"contains('{substring}')", check_fn=check)


def not_contains(substring: str) -> Constraint:
    """Output must NOT contain the given substring."""
    def check(output: str) -> tuple[bool, float, str]:
        if substring not in output:
            return True, 0.99, f"Does not contain '{substring}'"
        return False, 0.0, f"Unwanted '{substring}' found"
    return Constraint(name=f"not_contains('{substring}')", check_fn=check)


def valid_json() -> Constraint:
    """Output must be valid JSON."""
    def check(output: str) -> tuple[bool, float, str]:
        try:
            json.loads(output)
            return True, 0.99, "Valid JSON"
        except json.JSONDecodeError as e:
            return False, 0.0, f"Invalid JSON: {e}"
    return Constraint(name="valid_json", check_fn=check)


def matches_schema(schema: dict) -> Constraint:
    """Output must be valid JSON matching the given schema (key presence check)."""
    def check(output: str) -> tuple[bool, float, str]:
        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            return False, 0.0, f"Invalid JSON: {e}"

        if not isinstance(data, dict):
            return False, 0.1, "JSON is not an object"

        required_keys = set(schema.get("required", schema.get("properties", {}).keys()))
        present = required_keys & set(data.keys())
        missing = required_keys - set(data.keys())

        if not missing:
            return True, 0.99, "All required keys present"

        coverage = len(present) / max(len(required_keys), 1)
        return False, coverage * 0.9, f"Missing keys: {missing}"

    return Constraint(name="matches_schema", check_fn=check)


def custom(name: str, fn: Callable[[str], bool], fail_msg: str = "Custom check failed") -> Constraint:
    """Create a constraint from any boolean function."""
    def check(output: str) -> tuple[bool, float, str]:
        result = fn(output)
        if result:
            return True, 0.99, f"{name}: passed"
        return False, 0.0, f"{name}: {fail_msg}"
    return Constraint(name=name, check_fn=check)


# ---------------------------------------------------------------------------
# Core Engine
# ---------------------------------------------------------------------------

def ai_assert(
    prompt: str,
    constraints: list[Constraint],
    generate_fn: Callable[[str], str],
    max_retries: int = 3,
    feedback_fn: Optional[Callable[[str, list[CheckResult]], str]] = None,
) -> AiAssertResult:
    """
    Generate AI output, check constraints, retry with feedback if needed.

    Args:
        prompt: The prompt to send to the AI
        constraints: List of constraints to check
        generate_fn: Function that takes a prompt and returns AI output
        max_retries: Maximum number of retry attempts
        feedback_fn: Optional function to build feedback prompt from output + failed checks.
                     Default: appends failure messages to original prompt.

    Returns:
        AiAssertResult with output, pass/fail, checks, attempt count, and history.
    """
    if not constraints:
        output = generate_fn(prompt)
        return AiAssertResult(output=output, passed=True, attempts=1)

    current_prompt = prompt
    history = []

    for attempt in range(1, max_retries + 2):  # +2: range exclusive + 1 for initial
        output = generate_fn(current_prompt)

        # Check all constraints (multiplicative gate — all must pass)
        checks = [c.check(output) for c in constraints]
        all_passed = all(c.passed for c in checks)

        history.append({
            "attempt": attempt,
            "output": output,
            "checks": [(c.name, c.passed, c.score, c.message) for c in checks],
            "all_passed": all_passed,
        })

        if all_passed:
            return AiAssertResult(
                output=output,
                passed=True,
                checks=checks,
                attempts=attempt,
                history=history,
            )

        # If not all passed and we have retries left, build feedback
        if attempt <= max_retries:
            failed = [c for c in checks if not c.passed]
            if feedback_fn:
                current_prompt = feedback_fn(output, failed)
            else:
                # Default feedback: append failure info
                failure_info = "\n".join(
                    f"- {c.name}: {c.message}" for c in failed
                )
                current_prompt = (
                    f"{prompt}\n\n"
                    f"[CORRECTION NEEDED] Your previous output failed these checks:\n"
                    f"{failure_info}\n"
                    f"Previous output was: {output[:200]}\n"
                    f"Please fix and try again."
                )

    # All retries exhausted
    return AiAssertResult(
        output=output,
        passed=False,
        checks=checks,
        attempts=attempt,
        history=history,
    )


# ---------------------------------------------------------------------------
# Decorator API
# ---------------------------------------------------------------------------

def reliable(
    constraints: list[Constraint],
    max_retries: int = 3,
    feedback_fn: Optional[Callable] = None,
):
    """
    Decorator that wraps an AI generation function with constraint checking.

    Usage:
        @reliable(constraints=[max_length(100), valid_json()])
        def my_ai_fn(prompt: str) -> str:
            return call_llm(prompt)

        result = my_ai_fn("Generate a JSON response")
        # result is AiAssertResult
    """
    def decorator(fn: Callable[[str], str]):
        @functools.wraps(fn)
        def wrapper(prompt: str, **kwargs) -> AiAssertResult:
            return ai_assert(
                prompt=prompt,
                constraints=constraints,
                generate_fn=fn,
                max_retries=max_retries,
                feedback_fn=feedback_fn,
            )
        return wrapper
    return decorator
