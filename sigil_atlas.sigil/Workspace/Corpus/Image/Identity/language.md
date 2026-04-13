---
status: idea
---

# Identity

What identifies a given @image across the system. A content-hash derived from the image bytes.

!unique: no two distinct images share the same identity.
!stable: the identity does not change when the file moves, is renamed, or its @source reconnects at a different path.
!derived: computed from file content, not assigned. The same file on two different drives produces the same identity.

Duplicate detection is automatic: re-ingesting an image already in the @corpus is a no-op.
