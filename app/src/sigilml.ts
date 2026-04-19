/**
 * SigilML text parser — inline boolean expressions over @thing atoms.
 *
 * The spec (SigilML/language.md) says the user normally writes SigilML
 * through Controls, not text. This module is the escape hatch: an inline
 * typed form so the user can state intent in words and see the
 * parenthesized parse tree, to verify precedence and peer-group scope.
 *
 * Grammar (standard boolean precedence: not > and > or):
 *
 *     expr    := or
 *     or      := and ('or' and)*
 *     and     := not ('and' not)*
 *     not     := 'not' not | primary
 *     primary := '(' expr ')' | ident+
 *
 * Multi-word atoms are allowed: consecutive non-keyword tokens form a
 * single @thing name (so "red car" is one atom, not two).
 *
 * Formatter emits one set of parens around every composite sub-expression,
 * matching the spec's "peer-group = one set of parentheses" rule. The
 * top-level composite is the one exception — no outer parens, they add
 * nothing. `not X` is bare when X is an atom, `not(X)` when X is composite.
 */

import type { AndNode, Expression, NotNode, OrNode, ThingAtom } from "./relevance";

const KEYWORDS = new Set(["and", "or", "not", "(", ")"]);

export class SigilMLParseError extends Error {
  constructor(message: string, public position: number) {
    super(message);
    this.name = "SigilMLParseError";
  }
}

// --- Tokenizer ------------------------------------------------------------

interface Token {
  kind: "word" | "and" | "or" | "not" | "lparen" | "rparen";
  text: string;
  pos: number;
}

function tokenize(input: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  while (i < input.length) {
    const ch = input[i];
    if (/\s/.test(ch)) { i++; continue; }
    if (ch === "(") { tokens.push({ kind: "lparen", text: "(", pos: i }); i++; continue; }
    if (ch === ")") { tokens.push({ kind: "rparen", text: ")", pos: i }); i++; continue; }
    // A word runs until whitespace or a paren. Keyword matching is case-
    // insensitive and whole-token.
    const start = i;
    while (i < input.length && !/\s/.test(input[i]) && input[i] !== "(" && input[i] !== ")") {
      i++;
    }
    const text = input.slice(start, i);
    const lower = text.toLowerCase();
    if (lower === "and" || lower === "or" || lower === "not") {
      tokens.push({ kind: lower as Token["kind"], text: lower, pos: start });
    } else {
      tokens.push({ kind: "word", text, pos: start });
    }
  }
  return tokens;
}

// --- Parser ---------------------------------------------------------------

class Parser {
  private i = 0;
  constructor(private tokens: Token[], private source: string) {}

  parse(): Expression {
    if (this.tokens.length === 0) {
      throw new SigilMLParseError("empty expression", 0);
    }
    const expr = this.parseOr();
    if (this.i < this.tokens.length) {
      const t = this.tokens[this.i];
      throw new SigilMLParseError(`unexpected '${t.text}'`, t.pos);
    }
    return expr;
  }

  private peek(): Token | undefined { return this.tokens[this.i]; }

  private parseOr(): Expression {
    const first = this.parseAnd();
    const children: Expression[] = [first];
    while (this.peek()?.kind === "or") {
      this.i++;
      children.push(this.parseAnd());
    }
    return children.length === 1 ? first : ({ type: "or", children } as OrNode);
  }

  private parseAnd(): Expression {
    const first = this.parseNot();
    const children: Expression[] = [first];
    while (this.peek()?.kind === "and") {
      this.i++;
      children.push(this.parseNot());
    }
    return children.length === 1 ? first : ({ type: "and", children } as AndNode);
  }

  private parseNot(): Expression {
    if (this.peek()?.kind === "not") {
      this.i++;
      const child = this.parseNot();
      return { type: "not", child } as NotNode;
    }
    return this.parsePrimary();
  }

  private parsePrimary(): Expression {
    const t = this.peek();
    if (!t) throw new SigilMLParseError("unexpected end of expression", this.source.length);
    if (t.kind === "lparen") {
      this.i++;
      const inner = this.parseOr();
      const close = this.peek();
      if (close?.kind !== "rparen") {
        throw new SigilMLParseError("missing ')'", close ? close.pos : this.source.length);
      }
      this.i++;
      return inner;
    }
    if (t.kind !== "word") {
      throw new SigilMLParseError(`unexpected '${t.text}'`, t.pos);
    }
    // Greedy: consecutive word tokens form one multi-word atom.
    const parts: string[] = [];
    while (this.peek()?.kind === "word") {
      parts.push(this.tokens[this.i].text);
      this.i++;
    }
    return { type: "thing", name: parts.join(" ") } as ThingAtom;
  }
}

export function parseSigilML(input: string): Expression {
  const tokens = tokenize(input);
  return new Parser(tokens, input).parse();
}

// --- Formatter ------------------------------------------------------------

/**
 * Render an Expression as SigilML text with explicit peer-group parens.
 *
 * Rule: every AND/OR peer-group wraps in `(...)` EXCEPT the top-level one
 * (outer parens add nothing). `not X` is bare when X is an atom, `not(X)`
 * (no space) when X is composite — matches the spec's tight-binding NOT.
 */
export function formatSigilML(expr: Expression): string {
  return render(expr, true);
}

function render(expr: Expression, isTop: boolean): string {
  switch (expr.type) {
    case "thing":
      return expr.name;
    case "target_image":
      return `<image ${expr.image_id.slice(0, 8)}>`;
    case "contrast":
      return `<contrast ${expr.pole_a} vs ${expr.pole_b}>`;
    case "range":
      return `<range ${expr.dimension}>`;
    case "not": {
      const inner = expr.child;
      if (inner.type === "and" || inner.type === "or") {
        // `not(...)` supplies the peer-group parens; render the child as
        // top so it doesn't wrap a second time.
        return `not(${render(inner, true)})`;
      }
      return `not ${render(inner, false)}`;
    }
    case "and":
    case "or": {
      const op = expr.type === "and" ? " and " : " or ";
      const body = expr.children.map((c) => render(c, false)).join(op);
      return isTop ? body : `(${body})`;
    }
  }
}

// --- Heuristic: is this input a SigilML expression or a bare thing name? --

/**
 * True if the input looks like a boolean expression (contains any of the
 * keywords `and`/`or`/`not` as whole tokens, or parentheses). False means
 * it should be treated as a single bare thing name — the pre-existing
 * behaviour of the Attract input.
 */
export function isExpression(input: string): boolean {
  for (const t of tokenize(input)) {
    if (t.kind !== "word") return true;
  }
  return false;
}
