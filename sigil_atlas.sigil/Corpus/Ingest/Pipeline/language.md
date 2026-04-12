---
status: idea
---

# Pipeline

A !streaming, !cancellable and !resumable pipeline that coordinates all of its @stages in as !parallel manner as possible.

Responsibilities:

- reading the files from @source
- #builds-previews-and-thumbnails to feed embedding and UI
- #extracts-metadata to be available
- #embeds the images so we can look at them in @ContrastSpace
- #reports-progress for each of the @stages so I can see it in UI in real time
- I can #cancel it gracefully and can #resume it from that point.

