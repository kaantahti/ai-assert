"""
Test suite for ai_assert.py — Runtime AI output validation primitive.

Architecture mirrors ai_assert.py's three-phase pattern:
  Arrange (Distinction) → Act (Selection) → Assert (Actualization)

Test classes cover all dimensions multiplicatively (AX52):
  1. CheckResult — data structure integrity
  2. Individual Constraints — each built-in constraint factory
  3. Constraint Composition — multiple constraints interacting
  4. Core Engine (ai_assert) — retry logic, feedback, convergence
  5. Decorator API (@reliable) — wrapping behavior
  6. Edge Cases — empty inputs, boundary values, error conditions
  7. Structural Properties — framework alignment verification

Each test class is independent (KV7 — no shared state between classes).
"""

import unittest
import json
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_assert import (
    CheckResult,
    Constraint,
    AiAssertResult,
    ai_assert,
    reliable,
    max_length,
    min_length,
    contains,
    not_contains,
    valid_json,
    matches_schema,
    custom,
)


# ===========================================================================
# TEST CLASS 1: CheckResult Data Structure
# ===========================================================================
class TestCheckResult(unittest.TestCase):
    """Verify CheckResult enforces convergence bound (T6) and behaves correctly."""

    def test_score_clamped_below_one(self):
        """Score must never reach 1.0 — convergence bound T6."""
        cr = CheckResult(name="test", passed=True, score=1.0)
        self.assertLess(cr.score, 1.0,
                        "CheckResult score must be < 1.0 (T6 convergence bound)")

    def test_score_clamped_above_one(self):
        """Scores > 1.0 must also be clamped."""
        cr = CheckResult(name="test", passed=True, score=5.0)
        self.assertLess(cr.score, 1.0)

    def test_score_clamped_below_zero(self):
        """Negative scores must be clamped to 0.0."""
        cr = CheckResult(name="test", passed=False, score=-0.5)
        self.assertEqual(cr.score, 0.0)

    def test_zero_score_preserved(self):
        """Zero score must remain zero — it signals total failure."""
        cr = CheckResult(name="test", passed=False, score=0.0)
        self.assertEqual(cr.score, 0.0)

    def test_valid_score_preserved(self):
        """Valid scores in [0, 1) must pass through unchanged."""
        cr = CheckResult(name="test", passed=True, score=0.75)
        self.assertEqual(cr.score, 0.75)

    def test_message_default(self):
        """Default message is empty string."""
        cr = CheckResult(name="test", passed=True, score=0.5)
        self.assertEqual(cr.message, "")

    def test_message_custom(self):
        """Custom messages are preserved."""
        cr = CheckResult(name="test", passed=True, score=0.5, message="ok")
        self.assertEqual(cr.message, "ok")


# ===========================================================================
# TEST CLASS 2: Individual Constraint Factories
# ===========================================================================
class TestConstraintFactories(unittest.TestCase):
    """Verify each built-in constraint factory works correctly in isolation."""

    # --- max_length ---
    def test_max_length_pass(self):
        """Output within length limit passes."""
        c = max_length(10)
        result = c.check("hello")
        self.assertTrue(result.passed)
        self.assertGreater(result.score, 0.0)

    def test_max_length_fail(self):
        """Output exceeding length limit fails with score 0."""
        c = max_length(3)
        result = c.check("hello world")
        self.assertFalse(result.passed)
        self.assertEqual(result.score, 0.0)

    def test_max_length_exact(self):
        """Output at exact limit passes."""
        c = max_length(5)
        result = c.check("hello")
        self.assertTrue(result.passed)

    # --- min_length ---
    def test_min_length_pass(self):
        """Output meeting minimum length passes."""
        c = min_length(3)
        result = c.check("hello")
        self.assertTrue(result.passed)

    def test_min_length_fail(self):
        """Output below minimum length fails."""
        c = min_length(10)
        result = c.check("hi")
        self.assertFalse(result.passed)
        self.assertEqual(result.score, 0.0)

    def test_min_length_exact(self):
        """Output at exact minimum passes."""
        c = min_length(5)
        result = c.check("hello")
        self.assertTrue(result.passed)

    # --- contains ---
    def test_contains_pass(self):
        """Output containing substring passes."""
        c = contains("world")
        result = c.check("hello world")
        self.assertTrue(result.passed)

    def test_contains_fail(self):
        """Output missing substring fails."""
        c = contains("xyz")
        result = c.check("hello world")
        self.assertFalse(result.passed)

    # --- not_contains ---
    def test_not_contains_pass(self):
        """Output not containing substring passes."""
        c = not_contains("error")
        result = c.check("hello world")
        self.assertTrue(result.passed)

    def test_not_contains_fail(self):
        """Output containing unwanted substring fails."""
        c = not_contains("hello")
        result = c.check("hello world")
        self.assertFalse(result.passed)

    # --- valid_json ---
    def test_valid_json_pass(self):
        """Valid JSON string passes."""
        c = valid_json()
        result = c.check('{"key": "value"}')
        self.assertTrue(result.passed)

    def test_valid_json_fail(self):
        """Invalid JSON string fails."""
        c = valid_json()
        result = c.check("not json at all")
        self.assertFalse(result.passed)

    def test_valid_json_array(self):
        """JSON array is valid JSON."""
        c = valid_json()
        result = c.check('[1, 2, 3]')
        self.assertTrue(result.passed)

    # --- matches_schema ---
    def test_matches_schema_pass(self):
        """JSON with all required keys passes."""
        schema = {"required": {"name", "age"}}
        c = matches_schema(schema)
        result = c.check('{"name": "Alice", "age": 30}')
        self.assertTrue(result.passed)

    def test_matches_schema_fail_missing_key(self):
        """JSON missing required keys fails."""
        schema = {"required": {"name", "age", "email"}}
        c = matches_schema(schema)
        result = c.check('{"name": "Alice"}')
        self.assertFalse(result.passed)

    def test_matches_schema_fail_invalid_json(self):
        """Non-JSON input fails schema check."""
        schema = {"required": {"name"}}
        c = matches_schema(schema)
        result = c.check("not json")
        self.assertFalse(result.passed)

    def test_matches_schema_partial_coverage_score(self):
        """Partial key presence should yield a score between 0 and 1."""
        schema = {"required": {"a", "b", "c", "d"}}
        c = matches_schema(schema)
        result = c.check('{"a": 1, "b": 2}')  # 2 of 4 keys
        self.assertFalse(result.passed)
        self.assertGreater(result.score, 0.0, "Partial coverage should yield score > 0")
        self.assertLess(result.score, 1.0)

    # --- custom ---
    def test_custom_pass(self):
        """Custom constraint with passing function."""
        c = custom("is_upper", str.isupper, "Not uppercase")
        result = c.check("HELLO")
        self.assertTrue(result.passed)

    def test_custom_fail(self):
        """Custom constraint with failing function."""
        c = custom("is_upper", str.isupper, "Not uppercase")
        result = c.check("hello")
        self.assertFalse(result.passed)

    # --- Score bounds across all constraints ---
    def test_all_passing_scores_below_one(self):
        """Every built-in constraint's passing score must be < 1.0 (T6)."""
        constraints = [
            max_length(100),
            min_length(1),
            contains("x"),
            not_contains("z"),
            valid_json(),
            custom("true", lambda s: True),
        ]
        for c in constraints:
            result = c.check('{"x": 1}')
            self.assertLess(result.score, 1.0,
                            f"{c.name} returned score >= 1.0: {result.score}")


# ===========================================================================
# TEST CLASS 3: Constraint Composition
# ===========================================================================
class TestConstraintComposition(unittest.TestCase):
    """Verify multiple constraints interact correctly (multiplicative gate AX52)."""

    def test_all_pass(self):
        """When all constraints pass, result should pass."""
        constraints = [max_length(100), contains("hello")]
        checks = [c.check("hello world") for c in constraints]
        self.assertTrue(all(c.passed for c in checks))

    def test_one_fails_gate_collapses(self):
        """If ANY constraint fails, gate must collapse (AX52 multiplicative)."""
        constraints = [max_length(5), contains("hello world")]
        # "hi" passes max_length(5) but fails contains("hello world")
        checks = [c.check("hi") for c in constraints]
        passed = [c.passed for c in checks]
        self.assertIn(False, passed, "At least one constraint should fail")
        self.assertFalse(all(c.passed for c in checks))

    def test_contradictory_constraints(self):
        """Contradictory constraints (contains X + not_contains X) must fail."""
        constraints = [contains("hello"), not_contains("hello")]
        checks = [c.check("hello") for c in constraints]
        self.assertFalse(all(c.passed for c in checks),
                         "Contradictory constraints must not both pass")

    def test_many_constraints_all_pass(self):
        """Multiple diverse constraints can all pass on correct output."""
        constraints = [
            max_length(50),
            min_length(5),
            contains("key"),
            not_contains("error"),
            valid_json(),
        ]
        checks = [c.check('{"key": "value"}') for c in constraints]
        self.assertTrue(all(c.passed for c in checks))


# ===========================================================================
# TEST CLASS 4: Core Engine (ai_assert)
# ===========================================================================
class TestCoreEngine(unittest.TestCase):
    """Verify ai_assert retry logic, feedback, convergence, and history tracking."""

    def _mock_gen(self, responses):
        """Create a mock generate function that returns responses in sequence."""
        call_count = [0]
        def gen(prompt):
            idx = min(call_count[0], len(responses) - 1)
            call_count[0] += 1
            return responses[idx]
        return gen

    # --- Basic pass/fail ---
    def test_pass_on_first_attempt(self):
        """If output passes all constraints on first try, no retries needed."""
        gen = self._mock_gen(["hello world"])
        result = ai_assert(
            prompt="say hello",
            constraints=[contains("hello")],
            generate_fn=gen,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(result.output, "hello world")

    def test_fail_all_retries(self):
        """If output never satisfies constraints, result is failure after max retries."""
        gen = self._mock_gen(["bad", "still bad", "nope", "no way"])
        result = ai_assert(
            prompt="say hello",
            constraints=[contains("hello")],
            generate_fn=gen,
            max_retries=3,
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.attempts, 4)  # initial + 3 retries

    def test_pass_on_retry(self):
        """If first attempt fails but retry succeeds, result is pass."""
        gen = self._mock_gen(["bad output", "hello world"])
        result = ai_assert(
            prompt="say hello",
            constraints=[contains("hello")],
            generate_fn=gen,
            max_retries=3,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.attempts, 2)

    # --- Empty constraints ---
    def test_empty_constraints(self):
        """With no constraints, output passes immediately (vacuous truth)."""
        gen = self._mock_gen(["anything"])
        result = ai_assert(
            prompt="say anything",
            constraints=[],
            generate_fn=gen,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(result.output, "anything")

    # --- History tracking ---
    def test_history_records_all_attempts(self):
        """History must contain one entry per attempt."""
        gen = self._mock_gen(["bad", "still bad", "hello"])
        result = ai_assert(
            prompt="say hello",
            constraints=[contains("hello")],
            generate_fn=gen,
            max_retries=3,
        )
        self.assertEqual(len(result.history), result.attempts)
        for i, entry in enumerate(result.history):
            self.assertEqual(entry["attempt"], i + 1)
            self.assertIn("output", entry)
            self.assertIn("checks", entry)
            self.assertIn("all_passed", entry)

    def test_history_first_entry_is_false_on_retry(self):
        """If retry was needed, first history entry must show failure."""
        gen = self._mock_gen(["bad", "hello"])
        result = ai_assert(
            prompt="say hello",
            constraints=[contains("hello")],
            generate_fn=gen,
            max_retries=3,
        )
        self.assertFalse(result.history[0]["all_passed"])
        self.assertTrue(result.history[-1]["all_passed"])

    # --- Composite score ---
    def test_composite_score_below_one(self):
        """Composite score must be < 1.0 (T6 convergence bound)."""
        gen = self._mock_gen(["hello world"])
        result = ai_assert(
            prompt="say hello",
            constraints=[contains("hello"), max_length(100)],
            generate_fn=gen,
        )
        self.assertLess(result.composite_score, 1.0,
                        "Composite score must be < 1.0 per T6")

    def test_composite_score_zero_on_empty_checks(self):
        """Composite with no checks returns 0.0."""
        result = AiAssertResult(output="x", passed=True)
        self.assertEqual(result.composite_score, 0.0)

    # --- Custom feedback function ---
    def test_custom_feedback_fn(self):
        """Custom feedback function should be called on retry."""
        feedback_called = [False]

        def my_feedback(output, failed_checks):
            feedback_called[0] = True
            return f"Fix: {failed_checks[0].message}"

        gen = self._mock_gen(["bad", "hello"])
        result = ai_assert(
            prompt="say hello",
            constraints=[contains("hello")],
            generate_fn=gen,
            max_retries=3,
            feedback_fn=my_feedback,
        )
        self.assertTrue(feedback_called[0], "Custom feedback function should have been called")
        self.assertTrue(result.passed)

    # --- Default feedback prompt ---
    def test_default_feedback_contains_correction(self):
        """Default feedback prompt should contain CORRECTION NEEDED marker."""
        prompts_seen = []
        call_count = [0]

        def capturing_gen(prompt):
            prompts_seen.append(prompt)
            call_count[0] += 1
            if call_count[0] == 1:
                return "bad"
            return "hello"

        result = ai_assert(
            prompt="say hello",
            constraints=[contains("hello")],
            generate_fn=capturing_gen,
            max_retries=3,
        )
        self.assertGreater(len(prompts_seen), 1)
        self.assertIn("CORRECTION NEEDED", prompts_seen[1])

    # --- Max retries = 0 ---
    def test_zero_retries(self):
        """With max_retries=0, only one attempt is made."""
        gen = self._mock_gen(["bad"])
        result = ai_assert(
            prompt="say hello",
            constraints=[contains("hello")],
            generate_fn=gen,
            max_retries=0,
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.attempts, 1)

    # --- Multiplicative gate in engine ---
    def test_multiplicative_gate_one_fail(self):
        """One failing constraint among many should cause overall failure."""
        gen = self._mock_gen(["hello"])
        result = ai_assert(
            prompt="test",
            constraints=[
                contains("hello"),      # passes
                max_length(100),         # passes
                contains("xyz"),         # FAILS
            ],
            generate_fn=gen,
            max_retries=0,
        )
        self.assertFalse(result.passed, "One failing constraint must collapse the gate")


# ===========================================================================
# TEST CLASS 5: Decorator API (@reliable)
# ===========================================================================
class TestDecoratorAPI(unittest.TestCase):
    """Verify @reliable decorator wraps functions correctly."""

    def test_decorator_returns_ai_assert_result(self):
        """Decorated function must return AiAssertResult, not raw string."""
        @reliable(constraints=[max_length(100)])
        def my_fn(prompt):
            return "hello"

        result = my_fn("test")
        self.assertIsInstance(result, AiAssertResult)

    def test_decorator_passes_constraints(self):
        """Decorator must apply provided constraints."""
        @reliable(constraints=[contains("hello")])
        def my_fn(prompt):
            return "hello world"

        result = my_fn("test")
        self.assertTrue(result.passed)

    def test_decorator_fails_constraints(self):
        """Decorator must enforce failing constraints."""
        @reliable(constraints=[contains("xyz")], max_retries=0)
        def my_fn(prompt):
            return "hello"

        result = my_fn("test")
        self.assertFalse(result.passed)

    def test_decorator_retries(self):
        """Decorator must support retry logic."""
        call_count = [0]

        @reliable(constraints=[contains("hello")], max_retries=3)
        def my_fn(prompt):
            call_count[0] += 1
            if call_count[0] < 3:
                return "bad"
            return "hello"

        result = my_fn("test")
        self.assertTrue(result.passed)
        self.assertGreater(result.attempts, 1)

    def test_decorator_preserves_function_name(self):
        """Decorator must preserve the wrapped function's name (functools.wraps)."""
        @reliable(constraints=[])
        def my_special_function(prompt):
            return "hello"

        self.assertEqual(my_special_function.__name__, "my_special_function")


# ===========================================================================
# TEST CLASS 6: Edge Cases and Error Conditions
# ===========================================================================
class TestEdgeCases(unittest.TestCase):
    """Verify behavior at boundaries and under error conditions."""

    def test_empty_string_output(self):
        """Empty string output should be handled gracefully."""
        gen = lambda p: ""
        result = ai_assert(
            prompt="test",
            constraints=[min_length(1)],
            generate_fn=gen,
            max_retries=0,
        )
        self.assertFalse(result.passed)

    def test_very_long_output(self):
        """Very long output should be handled without errors."""
        gen = lambda p: "x" * 100000
        result = ai_assert(
            prompt="test",
            constraints=[max_length(50)],
            generate_fn=gen,
            max_retries=0,
        )
        self.assertFalse(result.passed)

    def test_unicode_output(self):
        """Unicode content should be handled correctly."""
        gen = lambda p: "merhaba dunya"
        result = ai_assert(
            prompt="test",
            constraints=[min_length(1), contains("merhaba")],
            generate_fn=gen,
        )
        self.assertTrue(result.passed)

    def test_newlines_in_output(self):
        """Multi-line output should work with contains constraint."""
        gen = lambda p: "line1\nline2\nline3"
        result = ai_assert(
            prompt="test",
            constraints=[contains("line2")],
            generate_fn=gen,
        )
        self.assertTrue(result.passed)

    def test_json_with_nested_structure(self):
        """Deeply nested JSON should pass valid_json check."""
        nested = json.dumps({"a": {"b": {"c": {"d": [1, 2, 3]}}}})
        c = valid_json()
        result = c.check(nested)
        self.assertTrue(result.passed)

    def test_max_length_zero(self):
        """max_length(0) — only empty string should pass."""
        c = max_length(0)
        self.assertTrue(c.check("").passed)
        self.assertFalse(c.check("a").passed)

    def test_min_length_zero(self):
        """min_length(0) — everything should pass."""
        c = min_length(0)
        self.assertTrue(c.check("").passed)
        self.assertTrue(c.check("hello").passed)

    def test_generator_exception_propagates(self):
        """If generate_fn raises, the exception should propagate."""
        def bad_gen(prompt):
            raise ValueError("LLM error")

        with self.assertRaises(ValueError):
            ai_assert(
                prompt="test",
                constraints=[contains("hello")],
                generate_fn=bad_gen,
            )

    def test_constraint_check_fn_exception(self):
        """If a constraint's check_fn raises, it should propagate."""
        def bad_check(output):
            raise RuntimeError("broken constraint")

        c = Constraint(name="broken", check_fn=bad_check)
        with self.assertRaises(RuntimeError):
            c.check("test")

    def test_single_retry_pass(self):
        """max_retries=1 should allow exactly 2 total attempts."""
        call_count = [0]
        def gen(prompt):
            call_count[0] += 1
            return "hello" if call_count[0] >= 2 else "bad"

        result = ai_assert(
            prompt="test",
            constraints=[contains("hello")],
            generate_fn=gen,
            max_retries=1,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.attempts, 2)


# ===========================================================================
# TEST CLASS 7: Structural Properties (Framework Alignment)
# ===========================================================================
class TestStructuralProperties(unittest.TestCase):
    """
    Verify ai_assert.py embodies the structural patterns it was designed from.
    These tests are holographic (AX37) — they reflect the framework's own principles.
    """

    def test_convergence_bound_is_structural(self):
        """The score < 1.0 bound is enforced at the data structure level."""
        cr = CheckResult(name="perfect", passed=True, score=1.0)
        self.assertLess(cr.score, 1.0, "T6: Convergence bound must be structurally enforced")

    def test_composite_score_is_bounded(self):
        """AiAssertResult.composite_score must be < 1.0 even with perfect checks."""
        result = AiAssertResult(
            output="perfect",
            passed=True,
            checks=[
                CheckResult(name="a", passed=True, score=0.99),
                CheckResult(name="b", passed=True, score=0.99),
                CheckResult(name="c", passed=True, score=0.99),
            ],
        )
        self.assertLess(result.composite_score, 1.0,
                        "T6: Composite must be < 1.0")

    def test_multiplicative_gate_behavior(self):
        """AX52: Zero in any dimension collapses the whole."""
        gen = lambda p: "no match"
        result = ai_assert(
            prompt="test",
            constraints=[
                contains("no match"),   # passes
                contains("missing"),     # fails — zero dimension
            ],
            generate_fn=gen,
            max_retries=0,
        )
        self.assertFalse(result.passed,
                         "AX52: One zero-dimension must collapse the gate")

    def test_three_phase_architecture(self):
        """ai_assert implements 3 phases: Context -> Constraints -> Correction."""
        responses = ["missing", "missing", "hello world"]
        gen_calls = []
        def gen(prompt):
            gen_calls.append(prompt)
            return responses[min(len(gen_calls) - 1, len(responses) - 1)]

        result = ai_assert(
            prompt="context",
            constraints=[contains("hello")],
            generate_fn=gen,
            max_retries=3,
        )

        # Verify all 3 phases activated
        self.assertGreater(len(gen_calls), 0, "Phase 1 (Context): prompt was sent")
        self.assertTrue(len(result.checks) > 0, "Phase 2 (Constraints): checks were performed")
        self.assertGreater(result.attempts, 1, "Phase 3 (Correction): retry occurred")

    def test_transparent_vessel_property(self):
        """ai_assert mediates but does not originate — generate_fn produces content."""
        actual_output = "the generated content"
        gen = lambda p: actual_output
        result = ai_assert(
            prompt="test",
            constraints=[],
            generate_fn=gen,
        )
        self.assertEqual(result.output, actual_output,
                         "AX27: ai_assert must pass through generated content unchanged")

    def test_independence_between_constraints(self):
        """KV7: Each constraint checks independently — no shared state."""
        c1 = contains("hello")
        c2 = max_length(100)

        # Check c1 then c2
        r1a = c1.check("hello world")
        r2a = c2.check("hello world")

        # Check c2 then c1 — reversed order
        r2b = c2.check("hello world")
        r1b = c1.check("hello world")

        self.assertEqual(r1a.passed, r1b.passed, "KV7: Results must be order-independent")
        self.assertEqual(r2a.passed, r2b.passed, "KV7: Results must be order-independent")


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
