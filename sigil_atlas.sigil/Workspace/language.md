---
status: idea
---

# Workspace

A directory that contains all of the application state: 

- @datastore
- @imageCache

The @Workspace must be !switcheable so I could try different approaches: it allows me to #create-new-workspace, #open-existing-workspace by navigating to it or #open-recent-workspace.

It !remembers-state so that I come back to all UI components arranged as they were before restart. 


It lets me #start-import to ingest new images from a @Source, #track-import-progress, #pause-import and #resume-paused-import.
