# Decision Frame

Viewing: Corpus

## User

Inhabit this Sigil and let's think how we can converge it to be specified enough to build this prototype. The focus is on Explore. I definitely underdefined it. This is the most creative part of this project.

## Partner

Yes. Right now Explore names three modes, but it does not yet say what the user can do in each mode, what state Explore holds, or what must stay stable while moving. The missing center is that Explore is not just “ways of viewing,” it is the control loop by which attention moves, perception updates, and preference gets refined.

I think the fastest convergence is to define Explore as an interaction grammar. Each step should say: what the current frame is, what gesture or input changes it, what structure in the corpus that movement traverses, and what feedback tells me I’m getting warmer, colder, broader, narrower, more novel, or more beautiful. If we can specify those transitions, the prototype becomes buildable even if the deeper theory stays open.

My read is that taste-like is the real heart, and timelike and spacelike are support modes. Timelike gives narrated traversal through an ordered slice. Spacelike gives simultaneous comparison in a local neighborhood. Taste-like gives gradient-following through preference space, where my actions slowly deform the field itself by teaching the system what “more like this” means for me.

So I would tighten Explore around one invariant: every move must preserve orientation. After any gesture, I should know what changed, what stayed fixed, and how to get back. If that invariant holds, creative exploration can be wild without becoming disorienting.

Concretely, Explore seems to need a few missing sigils. It needs something like Viewpoint, which is the current position and orientation in corpus-space. It needs Focus, which is the current center of attention or selected image or neighborhood. It needs Lens, which is the active projection from high-dimensional structure into the frame. It needs NavigationGesture, which maps user action to motion in that structure. And it needs PreferenceField, which is the emergent taste-gradient inferred from my choices.

Then the three modes become easier to distinguish. Timelike means the lens privileges an ordering contrast and motion advances along it. Spacelike means the lens privileges neighborhood relations and motion pans, zooms, and re-centers within a local manifold. Taste-like means the lens privileges preference gradients and motion follows attraction, surprise, or contrast relative to my evolving taste field.

The prototype question is what can be faked first while preserving the right shape. I think you do not need full recursive sigil neighborhoods yet. You need a local neighborhood graph over images, a way to project that graph into the frame, a way to anchor on one image or cluster, and a way to record positive and negative signals from the user so the neighborhood ordering can reweight.

The best underdefinition to remove first is “what is a move.” If we specify that, the rest can stay soft a bit longer. A move could be things like advance, retreat, widen, narrow, drift toward, drift away, pin this contrast, pivot around this image, and promote this image to exemplar. Those are buildable interaction verbs.

Then each mode can share the same verbs but interpret them differently. In timelike, advance means next along the chosen direction. In spacelike, advance means deeper into the selected neighborhood scale. In taste-like, advance means toward the region with higher predicted preference. That gives you one coherent interface rather than three separate apps.

I also think Explore needs an explicit account of uncertainty. The creative part is not only finding what you like, but discovering what you don’t yet know you like. So the system probably needs a tension between exploit and explore. In your language, not just beauty but also productive surprise. Otherwise taste-like will collapse too quickly into recommendation sludge.

So maybe TasteLike should not be “perpendicular to what I find most beautiful.” That phrase is evocative, but hard to build. More buildable is: the frame presents a frontier of candidate images balancing predicted beauty, diversity, and uncertainty. Moving forward increases confidence in a local taste region, while sideways movement preserves appeal but changes contrasts, and zooming changes scale of similarity.

That suggests a very practical prototype rule. Every frame is a set of images chosen by a policy over three signals: relevance to current focus, novelty relative to what has just been seen, and estimated preference from user feedback. Those weights can be adjustable or implicit. That is enough to make Explore feel alive.

If you want, I can next help you rewrite the Explore sigil itself into a tighter domain language and propose its first invariants and affordances in your native style.