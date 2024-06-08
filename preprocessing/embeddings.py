from pathlib import Path
from time import perf_counter
import torch
from transformers import AutoProcessor, CLIPModel

from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
input_directory = "data/images/128"
models_ids = [
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-base-patch16",
]


def main():
    directory = Path(input_directory)
    imgs = directory.iterdir()
    imgs_paths = [img_path for img_path in imgs]

    for model_id in models_ids:
        process_img_batch(imgs_paths, model_id)

    print(f"Processed {len(imgs_paths) * len(models_ids)} embeddings")


def process_img_batch(imgs_paths: Path, model_id: str):
    try:
        print(f"Starting processig for model {model_id}")
        model = CLIPModel.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        for i, img_path in enumerate(imgs_paths):
            process_img(img_path, processor, model)
            if i % 250 == 0:
                print(model_id, round(i * 100 / len(imgs_paths)), "%")
        print(f"Finished processing for  model {model_id}")
    except Exception as e:
        print(f"failed to process with {model_id}: {e}")


def process_img(img_path: Path, processor: AutoProcessor, model: CLIPModel):
    try:
        img_id = img_path.stem
        img = Image.open(img_path, formats=("WEBP",))

        img_inputs = processor(images=img, return_tensors="pt")
        img_embedding = model.get_image_features(**img_inputs)

        file = Path(f"data/embeddings/{model.name_or_path.replace('/','-')}/{img_id}")
        file.write_bytes(img_embedding.detach().numpy().tobytes())

    except Exception as e:
        print(f"failed to process {img_path}: {e}")


if __name__ == "__main__":
    t0 = perf_counter()
    main()
    t1 = perf_counter()
    print(f"Executed in {t1-t0}s")
