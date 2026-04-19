"""SigilML text parser — inline boolean expressions over @thing atoms.

The spec (SigilML/language.md) says the user normally writes SigilML through
Controls, not text. This module is the escape hatch: an inline typed form so
the user can state intent in words and see the parenthesized parse tree, to
verify precedence and peer-group scope.

Grammar (standard boolean precedence: not > and > or):

    expr    := or
    or      := and ('or' and)*
    and     := not ('and' not)*
    not     := 'not' not | primary
    primary := '(' expr ')' | ident+

Multi-word atoms are allowed: consecutive non-keyword tokens form a single
@thing name (so "red car" is one atom, not two).

Formatter emits one set of parens around every composite sub-expression,
matching the spec's "peer-group = one set of parentheses" rule. The top-
level composite is the one exception -- outer parens add nothing. `not X`
is bare when X is an atom, `not(X)` (no space) when X is composite.

Mirrors app/src/sigilml.ts one-to-one so TypeScript and Python produce the
same AST and the same formatted output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from sigil_atlas.relevance_filter import And, Contrast, Expression, Not, Or, Range, Thing, TargetImage


# --- Exceptions -----------------------------------------------------------


class SigilMLParseError(ValueError):
    def __init__(self, message: str, position: int) -> None:
        super().__init__(f"{message} at position {position}")
        self.position = position


# --- Tokenizer ------------------------------------------------------------


@dataclass(frozen=True)
class _Token:
    kind: str  # 'word' | 'and' | 'or' | 'not' | 'lparen' | 'rparen'
    text: str
    pos: int


def _tokenize(source: str) -> list[_Token]:
    tokens: list[_Token] = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "(":
            tokens.append(_Token("lparen", "(", i))
            i += 1
            continue
        if ch == ")":
            tokens.append(_Token("rparen", ")", i))
            i += 1
            continue
        start = i
        while i < n and not source[i].isspace() and source[i] not in "()":
            i += 1
        text = source[start:i]
        lower = text.lower()
        if lower in ("and", "or", "not"):
            tokens.append(_Token(lower, lower, start))
        else:
            tokens.append(_Token("word", text, start))
    return tokens


# --- Parser ---------------------------------------------------------------


class _Parser:
    def __init__(self, tokens: list[_Token], source: str) -> None:
        self._tokens = tokens
        self._source = source
        self._i = 0

    def parse(self) -> Expression:
        if not self._tokens:
            raise SigilMLParseError("empty expression", 0)
        expr = self._parse_or()
        if self._i < len(self._tokens):
            t = self._tokens[self._i]
            raise SigilMLParseError(f"unexpected '{t.text}'", t.pos)
        return expr

    def _peek(self) -> _Token | None:
        return self._tokens[self._i] if self._i < len(self._tokens) else None

    def _parse_or(self) -> Expression:
        first = self._parse_and()
        children: list[Expression] = [first]
        while self._peek() is not None and self._peek().kind == "or":  # type: ignore[union-attr]
            self._i += 1
            children.append(self._parse_and())
        return first if len(children) == 1 else Or(tuple(children))

    def _parse_and(self) -> Expression:
        first = self._parse_not()
        children: list[Expression] = [first]
        while self._peek() is not None and self._peek().kind == "and":  # type: ignore[union-attr]
            self._i += 1
            children.append(self._parse_not())
        return first if len(children) == 1 else And(tuple(children))

    def _parse_not(self) -> Expression:
        tok = self._peek()
        if tok is not None and tok.kind == "not":
            self._i += 1
            return Not(self._parse_not())
        return self._parse_primary()

    def _parse_primary(self) -> Expression:
        tok = self._peek()
        if tok is None:
            raise SigilMLParseError("unexpected end of expression", len(self._source))
        if tok.kind == "lparen":
            self._i += 1
            inner = self._parse_or()
            close = self._peek()
            if close is None or close.kind != "rparen":
                raise SigilMLParseError("missing ')'", close.pos if close else len(self._source))
            self._i += 1
            return inner
        if tok.kind != "word":
            raise SigilMLParseError(f"unexpected '{tok.text}'", tok.pos)
        parts: list[str] = []
        while self._peek() is not None and self._peek().kind == "word":  # type: ignore[union-attr]
            parts.append(self._tokens[self._i].text)
            self._i += 1
        return Thing(name=" ".join(parts))


def parse(source: str) -> Expression:
    """Parse a SigilML text expression into an Expression AST."""
    return _Parser(_tokenize(source), source).parse()


# --- Formatter ------------------------------------------------------------


def format_expression(expr: Expression) -> str:
    """Render an Expression as SigilML text with explicit peer-group parens.

    Every AND/OR peer-group is wrapped in `(...)` except the top-level one.
    `not X` is bare when X is an atom, `not(X)` when X is composite.
    """
    return _render(expr, top=True)


def _render(expr: Expression, top: bool) -> str:
    if isinstance(expr, Thing):
        return expr.name
    if isinstance(expr, TargetImage):
        return f"<image {expr.image_id[:8]}>"
    if isinstance(expr, Contrast):
        return f"<contrast {expr.pole_a} vs {expr.pole_b}>"
    if isinstance(expr, Range):
        return f"<range {expr.dimension}>"
    if isinstance(expr, Not):
        inner = expr.child
        if isinstance(inner, (And, Or)):
            # `not(...)` supplies the peer-group parens; render the child
            # as top so it doesn't wrap a second time.
            return f"not({_render(inner, top=True)})"
        return f"not {_render(inner, top=False)}"
    if isinstance(expr, (And, Or)):
        op = " and " if isinstance(expr, And) else " or "
        body = op.join(_render(c, top=False) for c in expr.children)
        return body if top else f"({body})"
    raise TypeError(f"Not an Expression: {type(expr).__name__}")


# --- Heuristic ------------------------------------------------------------


def is_expression(source: str) -> bool:
    """True if the input contains SigilML keywords or parens; false means
    treat the whole string as a single bare thing name.
    """
    return any(tok.kind != "word" for tok in _tokenize(source))
