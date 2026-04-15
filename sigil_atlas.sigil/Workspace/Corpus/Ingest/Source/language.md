---
status: idea
---

# Source

The source of images. It is !disconnectable, so we should cache previews in our @Workspace to be able to use the app when it's unavailable.

It is !uniquely-identified by #Location.

The first kind we support is @Folder on a file system. Or, it references a @URL. I can load @images from either @Folder or @URL.