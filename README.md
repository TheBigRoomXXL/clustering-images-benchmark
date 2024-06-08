# Real-Time image clustering

I am used to clustering images with embeddings and ward linkage but it's not 
ideal to run on low-compute devices such as cheap VM, laptop or 
even microcontrollers.

So instead I want to try with cheaper representation of images to reduce the
memory footprint and speedup the preproccessing and linkage steps.

Specifically I will experiment the following variation:
- CLIP patch 16 embedding with cosine distance (baseline)
- CLIP patch 32 embedding with cosine distance
- resized image with  hamming ditance
- aHash with hamming ditance
- pHash with hamming ditance
- dHash with hamming ditance

I will experiment using the [Flickr Image dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset?ref=hackernoon.com)


## Preprocessing the data set

First download my dataset and unzip it in `data/images/base/`. Then I resizing 
every images to a 128x128 webp. This will be this images used to calculates the
hash.

```bash
parallel convert {} -define webp:lossless=false -resize 8x8\! -quality 50% "data/images/8/{/.}.webp" ::: data/images/base/*

parallel -j 6 convert {} -define webp:lossless=false -resize 128x128\! -quality 50% "data/images/128/{/.}.webp" ::: data/images/base/*
```

Then I execute hash.py to compute all the hash.
