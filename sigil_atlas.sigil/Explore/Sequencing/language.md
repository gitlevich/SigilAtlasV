---
status: idea
---

# Sequencing

The ordered series derived from the current @slice when a mode needs one.

It answers: in what sequence should these @images be traversed? The sequence may come from capture date, a selected @contrastControl, or another ordering axis from the current @relevanceFilter.

@timelike renders this sequence as synced wrapped @strips. Other modes may ignore it or borrow it for secondary structure.

When sequencing changes, layout reflows smoothly but the underlying @slice stays the same.
