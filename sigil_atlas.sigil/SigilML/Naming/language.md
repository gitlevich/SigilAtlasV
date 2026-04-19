---
status: idea
---

# Naming

A graph-edit speech-act in @SigilML. `Name(sigil, as: role)` pronounced by a @POV from inside its @self installs a @child-edge in @self under the given @Role and a reciprocal edge in the named @sigil drawn from the role-taxonomy's pairings table. The return value is the named @sigil, so naming composes with expression operators — `Attract(Name(mama, as: мама))` names my mama under role `мама` and immediately attracts her.

The operator `Name` differs from the pattern @Sigil/Name. The operator pronounces; the pattern is what gets pronounced. Operator-`Name` installs pattern-@Name at a position (a @Role) in a @self.

Naming creates. If the first argument is a @sigil that does not yet exist, `Name` brings it into being — a mood I have just noticed, a place in my own lexicon, a person I am meeting — with the given @Name-pattern (optional, generated if omitted) at the named @Role. If the @sigil already exists, `Name` installs the @Role-edge without re-creating.

Providers assist, do not speak. @Face recognition, CLIP, and named-entity recognition offer candidate names from external taxonomies; the speaker accepts, rejects, corrects, or ignores. Manual naming is first-class. A @SigilML expression resolving to a @sigil is a third source.

The edge installed is reciprocal but asymmetric in label. My @self gains a door to mama under `мама`; her @self holds me under `сын`. One edge, two lexical projections. If the role-taxonomy declares no reciprocal, the other side holds a generic @Neighbor-edge until that side names it.

Retraction is another speech-act, not a deletion. `Retract(role-edge)` ends the edge from the speaker's side and marks the reciprocal as unilaterally severed on the other side. The trace is preserved; past namings are part of the @sigil's history, not undone.
