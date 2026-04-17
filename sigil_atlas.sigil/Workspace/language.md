---
status: idea
---

# Workspace

A directory that contains all of the application state for one way of working. Everything is scoped to a workspace: the @datastore, the @imageCache, the user-authored @sigils that define contexts, and the private experiences — both the user's and the recorded experience of the design partner the user entangles with. The authored spec is the medium of that entanglement; it carries structure and narrative together.

The @Workspace must be !switcheable so I could try different approaches: it allows me to #create-new-workspace, #open-existing-workspace by navigating to it, or #open-recent-workspace.

It !remembers-state so that I come back to all UI components arranged as they were before restart.

It lets me #start-import to ingest new @images from a @Source, #track-import-progress, #pause-import, and #resume-paused-import.
