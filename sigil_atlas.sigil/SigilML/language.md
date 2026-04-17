---
status: idea
---

# SigilML

Sigil Manipulation Language. The expression language for every transform the user performs on the @corpus. There is no separate grammar — the @sigil tree IS the syntax. The top token is a @sigil whose #affordances enumerate the possible next tokens; each subsequent @sigil's affordances similarly constrain what can follow. Completion, validation, and documentation come from the affordances of the current token.

Primitive operands — atoms: a @thing (membership), a @contrast range (a range between two sibling @things), a @TargetImage (a specific image).

Operators come in two kinds. Boolean, for composing predicates: unary `Not(expr)`, n-ary `And(exprs)`, n-ary `Or(exprs)`. Role-giving, for wrapping atoms so they take on a semantic role in an arrangement: `Attract(atom)` makes an atom a gravitational center, `Bandpass(contrast, min, max)` turns a contrast into a filter on a range, `OrderBy(contrast | time)` chooses a projection axis.

`And` and `Or` are n-ary. Their members are peers — sigils at the same @scope, grouped by one set of parentheses. A peer-group is itself a new sigil-term, usable at the parent scope alongside other peers. This is how same-scale composition is enforced structurally: you cannot accidentally combine terms at different scales, because bringing a term into a scope is an explicit act of wrapping it in a peer-group at that scope.

The user writes SigilML without seeing text. Every @Controls element on screen is populating a node of the current expression. Committing a row of tokens binds them as a peer-group. Opening a nested group descends into a child scope; closing it returns to the parent.

A @RelevanceFilter is a SigilML expression of boolean predicates over atoms. A @SpaceLike arrangement is a SigilML expression with `Attract` operators. A @TimeLike arrangement is a SigilML expression with `Bandpass` and `OrderBy` operators. A saved view is a serialized SigilML expression. An exported sigil archive carries its expression. Two users who exchange a sigil exchange a program, not a query.
