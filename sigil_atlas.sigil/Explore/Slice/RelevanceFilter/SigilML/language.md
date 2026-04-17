---
status: idea
---

# SigilML

Sigil Manipulation Language. Expressed as a chain of @sigils. The top token is a @sigil whose #affordances enumerate the possible next tokens. Each subsequent @sigil's affordances similarly constrain what can follow. The chain grows in specificity until it points at the exact thing whose meaning it constrains in @contrastspace.

There is no separate grammar — the @sigil tree IS the syntax. Completion, validation, and documentation come from the affordances of the current token.

Primitive operands: a @thing (membership), a @contrast range (a range between @things). Combinators: AND, OR, NOT. A @RelevanceFilter is a SigilML expression. So is any compound @thing or compound @contrast — same chain-of-affordances mechanic composes them all.
