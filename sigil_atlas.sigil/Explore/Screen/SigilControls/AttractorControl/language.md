---
status: idea
---

# Attractor Control

This control lets me define @attractors to name things I want to see in the @slice. 

One @attractor !attracts-images-with-all-named-things, working as AND operation: `bees, beetles and birds` should attract @images containing all three.

@AttractorControl constrains the @slice by only including @images pulled by every configured @attractor. 

@AttractorControl is !semantic-only. Its state is !persistent: if I defined an @attractor, it remains until I #deleted it.