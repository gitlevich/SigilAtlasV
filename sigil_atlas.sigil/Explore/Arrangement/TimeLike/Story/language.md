---
status: idea
---

# Story

A story is a burst of @images close in time. Stories are recursive: several images during a walk, a few walks during the day, several days of shooting over a week.

At a given @timescale, images cluster by temporal proximity. The gap threshold determines what counts as "close" — minutes for a walk, hours for a day, days for a trip.

Stories nest: a day-story contains walk-stories. A trip-story contains day-stories. Zooming out in @timescale collapses fine stories into coarser ones.

## Detection

Given images sorted by capture_date and a gap threshold g:
1. Walk the sorted list. When the time gap between consecutive images exceeds g, start a new story.
2. Each story is a contiguous run of images with no gap > g.
3. Stories with fewer than 2 images merge into their nearest neighbor.
