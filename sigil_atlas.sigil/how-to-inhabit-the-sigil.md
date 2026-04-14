# Implementer's Guide to the Sigil Spec

## Structure

This spec is a tree of directories. Each directory is a **sigil** — a named concept with boundaries and affordances.

Every sigil has:
- `language.md` — what this sigil is, written in the vocabulary of its children
- `affordance-*.md` — what this sigil offers (actions, behaviors, features)
- `invariant-*.md` — what must always be true inside this sigil
- `affordance.order` — the intended reading order for affordances
- Child directories — deeper concepts that this sigil is composed of

## How to read it

Start at the root `language.md`. It gives the shape of the whole application in a few sentences using its children's names.

To implement a specific feature, read the path from root to that sigil. Each level adds context and narrows intent. The root tells you why. The middle levels tell you how the pieces relate. The leaf tells you what to build.

You don't need to read the entire tree. You need everything directly above what you're implementing (the ancestors) and everything directly below it (the children). Siblings give you context about what's nearby but aren't required.

## References

- `@name` references a sigil. It resolves by looking in the current scope, then outward through ancestors, then into imported ontologies under `Libs/`.
- `#name` references an affordance defined in the current sigil or an ancestor.
- `!name` references an invariant defined in the current sigil or an ancestor.

Plurals, case differences, and past tenses all resolve to the same sigil. The spec reads like English.

## Libs

The `Libs/` subtree contains imported ontologies — vocabulary from external domains (Attention Language, Ecological Psychology, Differential Geometry, McGilchrist). These provide the theoretical grounding. They are ambient scope: visible everywhere, but local definitions shadow them.

You don't implement Libs. You use their vocabulary to understand the spec's intent.

## What to implement

Affordances are your feature list. Each `affordance-*.md` describes one thing the user can do. Implement it.

Invariants are your test suite. Each `invariant-*.md` describes something that must never break. Test for it.

The `language.md` narrative shows the affordances working together. It's the acceptance test: if the narrative reads true in the running application, the sigil is correctly implemented.

## What not to do

Don't implement what isn't specified. If a sigil is empty or says "status: idea" with no affordances, skip it.

Don't add structure the spec doesn't call for. If the spec says one slider, build one slider.

Don't contradict an invariant to make implementation easier. If it says "never jumps," it means never.

## Working style

Pick one sigil to implement. Read its ancestors for context. Read its children for detail. Build it. Test the invariants. Move to the next.
