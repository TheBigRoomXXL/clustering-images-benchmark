from pathlib import Path
from time import perf_counter

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"

lazy = True

input_directory = Path("data/images/flowers-clean")


def main(directory: Path):
    imgs = [file for file in directory.iterdir()]
    imgs_paths = [img_path for img_path in imgs]

    n = 1000
    imgs_path_chunks = [imgs_paths[i : i + n] for i in range(0, len(imgs_paths), n)]

    model = AutoModel.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    for chunk in imgs_path_chunks:
        process_img_batch(model, processor, chunk)

    print(f"Processed {len(imgs_paths) } embeddings")


def process_img_batch(
    model: AutoModel, processor: AutoProcessor, imgs_paths: list[Path]
):
    save_files = [
        Path(f"data/effigy/{model.name_or_path.replace('/','-')}/{img.stem}")
        for img in imgs_paths
    ]

    if lazy:
        save_files = [f for f in save_files if not f.is_file()]
        if len(save_files) == 0:
            print("0 file to process")
            return

    imgs = [Image.open(img_path) for img_path in imgs_paths]
    inputs = processor(images=imgs, return_tensors="pt")

    with torch.no_grad():
        img_embeddings = model.get_image_features(pixel_values=inputs["pixel_values"])

    img_embeddings_np = img_embeddings.detach().numpy()

    for save_file, img_embedding_np in zip(save_files, img_embeddings_np):
        save_file.write_bytes(img_embedding_np.tobytes())

    print(f"processed {len(save_files)} images")


if __name__ == "__main__":
    t0 = perf_counter()
    main(input_directory)
    t1 = perf_counter()
    print(f"Executed in {t1-t0}s")
