---
status: idea
---

# Metadata

Information extracted from an @image file at @ingest time. Not derived from visual content — from file headers.

- **capture_date**: when the photo was taken. Primary sort key in @timelike mode.
- **camera, lens**: equipment used.
- **exposure**: aperture, shutter speed, ISO.
- **GPS**: location coordinates, if present.
- **pixel dimensions**: original width and height.

Source: EXIF, IPTC, XMP headers embedded in the file. Extracted once during the metadata @stage of the @pipeline.
