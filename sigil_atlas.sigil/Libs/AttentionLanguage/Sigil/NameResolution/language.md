---
status: external-ontology
---

# Name Resolution

Looking up a @Name by walking outward through enclosing @Scopes until a binding is found. The innermost binding wins.

The path: current @Scope first, then parent, then ancestors, then ambient (imported ontologies). If no @Scope in the chain contains the @Name, it is unresolved — a gap in the domain language.

In a @Sigil tree, a statement resolves by tracing a path through nested scopes. If every @Name in the statement resolves, the language is projectable at that point. Unresolved @Names are the spec's open questions.

Resolution applies equally to text and to experience. When an @Agent encounters a @Name, it resolves against prior encounters using the same scoping rules — recognition is name resolution applied to experience.
