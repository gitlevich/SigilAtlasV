# Ingest

The process of bringing images from external location into the @corpus. 

To ingest images from a @Source, we specify one and  @pipeline#run agains it, resulting in the side-effect of the @corpus getting populated. When it runs, the @pipeline:

- @pipeline#extracts-metadata to grab all available @metadata 
- @pipeline#builds-previews-and-thumbnails so we have something to embed and to show
- @pipeline#embeds so we have embeddings
