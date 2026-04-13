---
status: idea
---

# Stage

One step of the @pipeline. Each stage does one transformation on an @image being ingested:

- **Scan**: discover files from @source, determine what is new or changed.
- **Thumbnail**: generate micro and thumb previews, store in @imageCache.
- **Metadata**: extract EXIF and other @metadata from the file.
- **Embed**: run each @visionModel to produce @embeddings.

Stages run as independently as possible — thumbnail generation and metadata extraction happen in parallel. Embedding depends on having a thumbnail or the original file available.

Each stage reports its own @progressStatus.
