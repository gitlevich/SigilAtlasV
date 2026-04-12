---
status: idea
---

# Image Wrapping Algorithm

Rule: what is not named doesn't matter.

In the context of an @ImageSigil, we only care about some @MajorContrasts and @SigilProximity: those we explicitly named: what is in the @image.

For that, we ask CLIP these questions:

- what is the main @subject of this @image 
- what other @subjects appear in the @image
- where is this @image on contrasts like dark vs bright, textured vs smooth, complex vs simple, a contrast between any two opposite colors, etc.
- what describes the image in a sentence (caption)
- what @narrative does the image tell
- ...

In other words, we #characterize each @image in words of some @ontology we got to define. 

