# Decision Frame

Viewing: Corpus

## User

Well, about viewpoint and focus, I kind of thought about the frame being that. Think about the frame as a frame of a television, and what we show on the screen of that television is the thing that we need to find out. Imagine that I am using just arrow keys to navigate through my current frame. But the more interesting bit is because the corpus could be enormous, I'd like to basically reduce it to just a cellular structure—black and white, with white cells separated from each other by black boundaries—when I'm fully zoomed out. It's multi-multi-scale, so I will see structures within structures within structures.

This happens because we sliced this sigil along some plane. I want to control the orientation of this plane in the 3D space to which we reduced the embedding spaces, so there is a correlation between the image I will find beautiful and the cellular shape I see at the fullest zoom. And as I slowly zoom in, with every new generation, we add one more contrast.

## Partner

Yes, that helps a lot, because it means Frame is not just a display surface but the actual navigable viewport into a slice of the Sigil. So I would stop separating viewpoint from frame and instead say the Frame carries at least three hidden state variables: slice plane orientation, zoom generation, and in-frame cursor position, while arrow keys move the cursor or pan the visible region depending on scale.

The strongest convergence I hear is that Explore is really about traversing a recursively sliced cellular map, where the fully zoomed-out view is pure morphology before image identity. Then the key buildable question becomes: what exact rule turns a 3D slice through reduced corpus space into black boundaries and white cells, and what exact rule makes zooming reveal one additional contrast per generation without breaking orientation or the user’s sense that they are still inside the same structure.

I think the prototype should treat this as one invariant object across scale: the same slice plane, progressively refined. If you want, I can now help turn this into a crisp spec by defining Frame, SlicePlane, Generation, Cell, Boundary, and the exact semantics of arrow keys and zoom so the first version becomes implementable.