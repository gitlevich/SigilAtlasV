---
status: idea
---

# Folder

A @source backed by a directory on the filesystem. Its #location is an absolute path.

!movable: if the physical directory moves (external drive remounted at a different path, renamed), I update the location without re-ingesting. @Image @identity is by content hash, not path.

!disconnectable: inherited from @source. When the drive is unmounted, the @corpus still works at micro and thumb resolution from @imageCache. Full-resolution display waits for reconnection.

On @ingest, the @pipeline walks the folder recursively, discovering files that match accepted formats.
