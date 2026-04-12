---
status: external-ontology
---

# Scope

A region where @Names are visible. Every @Sigil introduces a scope. Within it, each @Name means exactly one thing.

Defining a word in a @Sigil binds that @Name to this scope. Children introduce nested scopes. A @Name bound in an inner scope @shadows any binding of the same @Name in an enclosing scope.

Ancestors remain in scope: @Names bound in enclosing @Sigils are visible unless shadowed. Imported ontologies are ambient scope — visible everywhere, as if bound at the outermost level.
