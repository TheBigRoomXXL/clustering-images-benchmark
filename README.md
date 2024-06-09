# Image clustering benchmark

I am used to clustering images with embeddings and ward linkage but it's not 
ideal to run on low-compute devices such as cheap VM, laptop or 
even microcontrollers.

So instead I want to try with cheaper representation of images to reduce the
memory footprint and speedup the preproccessing and linkage steps.

Specifically I will experiment the following variation:
- CLIP patch 16 embedding with cosine distance (baseline)
- CLIP patch 32 embedding with cosine distance
- aHash with hamming ditance
- pHash with hamming ditance
- dHash with hamming ditance
- colorHash with hamming distance

And various dataset size for 10 to 30k.

Because the end-goal is real-time image cluster, I would also like to try with a two step clustering using first k-mean and then the ward linkage. I think it 
will greatly improve the performance for large sample.

I will experiment using the [Flickr Image dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset?ref=hackernoon.com)


## Result

### aHsh, pHash, dHash, colorHash just doesn't seam to work for efficent clustering :-(

## Running the benchmark

### 1. Preprocessing the data set

First download my dataset and unzip it in `data/images/base/`. Then I resizing 
every images to a 128x128 webp. This will be this images used to calculates the
hash.

```bash
parallel -j 6 convert {} -define webp:lossless=false -resize 128x128\! -quality 50% "data/images/thumbnail/{/.}.webp" ::: data/images/base/*
```

Then I execute `hash.py` and `embeddingpy` to compute all the hash and embedding.

### 2. Execute the benchmark

```bash
python benchmark.py
```


