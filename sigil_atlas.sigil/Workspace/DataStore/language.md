---
status: idea
---

# Data Store

Where the @corpus database, @embedding vectors, and @imageCache live on disk. Contained within the workspace directory.

Local only — no cloud dependency. Survives @source disconnection: everything needed to browse at micro and thumb resolution is here. Full-resolution display requires the @source to be connected.

Implementation: SQLite for @metadata, @identity mappings, and source records. @Embedding vectors as numpy arrays or database blobs. Thumbnails as files under the @imageCache directory tree.
