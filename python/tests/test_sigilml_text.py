"""Tests for the SigilML text parser and formatter.

The round-trip from source -> parse -> format gives the user the
parenthesized form they can read off to verify precedence and scope.
"""

from __future__ import annotations

import pytest

from sigil_atlas.relevance_filter import And, Not, Or, Thing
from sigil_atlas.sigilml_text import (
    SigilMLParseError,
    format_expression,
    is_expression,
    parse,
)


def _round_trip(source: str) -> str:
    return format_expression(parse(source))


# --- Atoms and single-level composition ----------------------------------


def test_single_atom():
    assert parse("red") == Thing("red")
    assert _round_trip("red") == "red"


def test_multi_word_atom():
    assert parse("red car") == Thing("red car")
    assert _round_trip("red car") == "red car"


def test_and_pair():
    assert parse("red and blue") == And((Thing("red"), Thing("blue")))
    # Top-level peer-group omits outer parens (spec: one set, no redundancy).
    assert _round_trip("red and blue") == "red and blue"


def test_or_pair():
    assert parse("red or blue") == Or((Thing("red"), Thing("blue")))
    assert _round_trip("red or blue") == "red or blue"


def test_not_atom():
    assert parse("not red") == Not(Thing("red"))
    assert _round_trip("not red") == "not red"


# --- Precedence ----------------------------------------------------------


def test_not_binds_tighter_than_and():
    # not c and d  =>  (not c) and d
    expr = parse("not red and blue")
    assert expr == And((Not(Thing("red")), Thing("blue")))
    assert _round_trip("not red and blue") == "not red and blue"


def test_and_binds_tighter_than_or():
    # a and b or c  =>  (a and b) or c
    expr = parse("red and green or blue")
    assert expr == Or((And((Thing("red"), Thing("green"))), Thing("blue")))
    # Peer-group (a and b) is NOT top-level anymore, so it gets parens.
    assert _round_trip("red and green or blue") == "(red and green) or blue"


def test_chained_and_is_nary():
    # a and b and c  =>  And(a, b, c)  (single peer-group, not nested)
    expr = parse("red and green and blue")
    assert expr == And((Thing("red"), Thing("green"), Thing("blue")))
    assert _round_trip("red and green and blue") == "red and green and blue"


def test_chained_or_is_nary():
    expr = parse("red or green or blue")
    assert expr == Or((Thing("red"), Thing("green"), Thing("blue")))
    assert _round_trip("red or green or blue") == "red or green or blue"


# --- The user's example --------------------------------------------------


def test_user_example_renders_with_explicit_groups():
    # Standard precedence: not > and > or.
    #   red and yellow and orange and green and not white or blue
    # parses as
    #   ((red and yellow and orange and green and (not white)) or blue)
    # Top-level is Or; its first child is the And peer-group, so that group
    # gets parens. `not white` is bare (operand is an atom).
    src = "red and yellow and orange and green and not white or blue"
    formatted = _round_trip(src)
    assert formatted == "(red and yellow and orange and green and not white) or blue"


def test_not_over_parenthesised_or():
    # `not(white or blue)` is representable by explicit parens on the input.
    expr = parse("not (white or blue)")
    assert expr == Not(Or((Thing("white"), Thing("blue"))))
    assert _round_trip("not (white or blue)") == "not(white or blue)"


# --- Explicit parens alter grouping --------------------------------------


def test_explicit_parens_override_precedence():
    expr = parse("red and (green or blue)")
    assert expr == And((Thing("red"), Or((Thing("green"), Thing("blue")))))
    assert _round_trip("red and (green or blue)") == "red and (green or blue)"


def test_double_negation():
    expr = parse("not not red")
    assert expr == Not(Not(Thing("red")))
    assert _round_trip("not not red") == "not not red"


# --- Error cases ---------------------------------------------------------


def test_empty_string_raises():
    with pytest.raises(SigilMLParseError):
        parse("")


def test_missing_rparen_raises():
    with pytest.raises(SigilMLParseError):
        parse("red and (green or blue")


def test_trailing_operator_raises():
    with pytest.raises(SigilMLParseError):
        parse("red and")


def test_leading_operator_raises():
    with pytest.raises(SigilMLParseError):
        parse("and red")


# --- is_expression heuristic --------------------------------------------


def test_is_expression_true_for_keywords():
    assert is_expression("red and blue") is True
    assert is_expression("not red") is True
    assert is_expression("(red)") is True


def test_is_expression_false_for_bare_names():
    assert is_expression("red") is False
    assert is_expression("red car") is False
    assert is_expression("") is False
