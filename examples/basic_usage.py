"""
basic_usage.py — How to use ai_assert with different generators.

Three examples:
  1. Mock generator  — works immediately, no dependencies
  2. OpenAI          — requires: pip install openai + OPENAI_API_KEY
  3. Anthropic       — requires: pip install anthropic + ANTHROPIC_API_KEY

Run:
    python examples/basic_usage.py
"""

import sys
import os

# Add project root to path so we can import ai_assert
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_assert import (
    ai_assert,
    reliable,
    max_length,
    min_length,
    contains,
    not_contains,
    valid_json,
    matches_schema,
    custom,
    AiAssertResult,
)


# ---------------------------------------------------------------------------
# Example 1: Mock Generator (no dependencies needed)
# ---------------------------------------------------------------------------

def example_mock():
    """Demonstrate ai_assert with a simple mock that improves on retry."""
    print("=" * 60)
    print("Example 1: Mock Generator")
    print("=" * 60)

    call_count = 0

    def mock_generator(prompt: str) -> str:
        nonlocal call_count
        call_count += 1

        # First attempt: too short, missing keyword
        if call_count == 1:
            return "Hi"

        # Second attempt: has keyword but invalid JSON
        if call_count == 2:
            return '{"greeting": "Hello, World!", broken}'

        # Third attempt: correct
        return '{"greeting": "Hello, World!"}'

    result = ai_assert(
        prompt="Generate a JSON greeting that says Hello, World!",
        constraints=[
            valid_json(),
            min_length(10),
            contains("Hello"),
        ],
        generate_fn=mock_generator,
        max_retries=3,
    )

    print(f"  Passed:    {result.passed}")
    print(f"  Output:    {result.output}")
    print(f"  Attempts:  {result.attempts}")
    print(f"  Score:     {result.composite_score:.4f}")
    print(f"  Checks:")
    for check in result.checks:
        mark = "PASS" if check.passed else "FAIL"
        print(f"    - {check.name}: [{mark}] ({check.score:.2f}) {check.message}")
    print()


# ---------------------------------------------------------------------------
# Example 2: Decorator API with Mock
# ---------------------------------------------------------------------------

def example_decorator():
    """Demonstrate the @reliable decorator pattern."""
    print("=" * 60)
    print("Example 2: @reliable Decorator")
    print("=" * 60)

    @reliable(
        constraints=[
            max_length(200),
            min_length(20),
            not_contains("ERROR"),
            custom(
                name="starts_with_capital",
                fn=lambda s: len(s) > 0 and s[0].isupper(),
                fail_msg="Output must start with a capital letter",
            ),
        ],
        max_retries=2,
    )
    def generate_summary(prompt: str) -> str:
        """Simulates an AI that generates text summaries."""
        return "The quick brown fox jumps over the lazy dog. This is a summary of the requested content."

    result = generate_summary("Summarize the document.")

    print(f"  Passed:    {result.passed}")
    print(f"  Output:    {result.output[:80]}...")
    print(f"  Attempts:  {result.attempts}")
    print(f"  Score:     {result.composite_score:.4f}")
    print()


# ---------------------------------------------------------------------------
# Example 3: Schema Validation
# ---------------------------------------------------------------------------

def example_schema():
    """Demonstrate JSON schema validation with retry."""
    print("=" * 60)
    print("Example 3: Schema Validation")
    print("=" * 60)

    call_count = 0

    def json_generator(prompt: str) -> str:
        nonlocal call_count
        call_count += 1

        # First attempt: missing "age" field
        if call_count == 1:
            return '{"name": "Alice"}'

        # Second attempt: all fields present
        return '{"name": "Alice", "age": 30, "email": "alice@example.com"}'

    result = ai_assert(
        prompt="Generate a user profile as JSON with name, age, and email.",
        constraints=[
            valid_json(),
            matches_schema({"required": ["name", "age", "email"]}),
        ],
        generate_fn=json_generator,
        max_retries=2,
    )

    print(f"  Passed:    {result.passed}")
    print(f"  Output:    {result.output}")
    print(f"  Attempts:  {result.attempts}")
    print(f"  Score:     {result.composite_score:.4f}")
    print()

    # Show retry history
    print("  Retry history:")
    for entry in result.history:
        status = "✓" if entry["all_passed"] else "✗"
        print(f"    Attempt {entry['attempt']} {status}: {entry['output'][:50]}")
    print()


# ---------------------------------------------------------------------------
# Example 4: Custom Feedback Function
# ---------------------------------------------------------------------------

def example_custom_feedback():
    """Demonstrate a custom feedback function for smarter retries."""
    print("=" * 60)
    print("Example 4: Custom Feedback Function")
    print("=" * 60)

    call_count = 0

    def generator(prompt: str) -> str:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            return "the answer is 42"  # lowercase start
        return "The answer is 42"  # fixed

    def smart_feedback(output, failed_checks):
        """Build a targeted feedback prompt from failures."""
        issues = [f"[{c.name}] {c.message}" for c in failed_checks]
        return (
            f"Your output had issues:\n"
            + "\n".join(issues)
            + f"\n\nOriginal output: {output}\n"
            + "Please correct these specific issues."
        )

    result = ai_assert(
        prompt="What is the meaning of life?",
        constraints=[
            custom(
                "starts_uppercase",
                lambda s: s[0].isupper() if s else False,
                "Must start with uppercase",
            ),
            contains("42"),
        ],
        generate_fn=generator,
        max_retries=2,
        feedback_fn=smart_feedback,
    )

    print(f"  Passed:    {result.passed}")
    print(f"  Output:    {result.output}")
    print(f"  Attempts:  {result.attempts}")
    print()


# ---------------------------------------------------------------------------
# Example 5: OpenAI Integration (requires pip install openai)
# ---------------------------------------------------------------------------

def example_openai():
    """Demonstrate ai_assert with OpenAI API."""
    print("=" * 60)
    print("Example 5: OpenAI Integration")
    print("=" * 60)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("  Skipped — set OPENAI_API_KEY to run this example.")
        print()
        return

    try:
        from openai import OpenAI
    except ImportError:
        print("  Skipped — run: pip install openai")
        print()
        return

    client = OpenAI(api_key=api_key)

    def openai_generate(prompt: str) -> str:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return response.choices[0].message.content or ""

    result = ai_assert(
        prompt="Return a JSON object with keys 'city' and 'population'. Pick any city.",
        constraints=[
            valid_json(),
            matches_schema({"required": ["city", "population"]}),
            max_length(500),
        ],
        generate_fn=openai_generate,
        max_retries=2,
    )

    print(f"  Passed:    {result.passed}")
    print(f"  Output:    {result.output}")
    print(f"  Attempts:  {result.attempts}")
    print(f"  Score:     {result.composite_score:.4f}")
    print()


# ---------------------------------------------------------------------------
# Example 6: Anthropic Integration (requires pip install anthropic)
# ---------------------------------------------------------------------------

def example_anthropic():
    """Demonstrate ai_assert with Anthropic API."""
    print("=" * 60)
    print("Example 6: Anthropic Integration")
    print("=" * 60)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  Skipped — set ANTHROPIC_API_KEY to run this example.")
        print()
        return

    try:
        from anthropic import Anthropic
    except ImportError:
        print("  Skipped — run: pip install anthropic")
        print()
        return

    client = Anthropic(api_key=api_key)

    def anthropic_generate(prompt: str) -> str:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    result = ai_assert(
        prompt="Return ONLY a JSON object with keys 'language' and 'year_created'. Pick any programming language.",
        constraints=[
            valid_json(),
            matches_schema({"required": ["language", "year_created"]}),
            min_length(10),
        ],
        generate_fn=anthropic_generate,
        max_retries=2,
    )

    print(f"  Passed:    {result.passed}")
    print(f"  Output:    {result.output}")
    print(f"  Attempts:  {result.attempts}")
    print(f"  Score:     {result.composite_score:.4f}")
    print()


# ---------------------------------------------------------------------------
# Run all examples
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nai_assert — Basic Usage Examples\n")

    example_mock()
    example_decorator()
    example_schema()
    example_custom_feedback()
    example_openai()
    example_anthropic()

    print("Done.")
