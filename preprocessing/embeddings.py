from pathlib import Path
from time import perf_counter
import torch
from transformers import AutoProcessor, CLIPModel, AutoModel

from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
models_ids = [
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-base-patch16",
]
lazy = False


def main(directory: Path):
    imgs = directory.iterdir()
    imgs_paths = [img_path for img_path in imgs]

    for model_id in models_ids:
        process_img_batch(imgs_paths, model_id)

    print(f"Processed {len(imgs_paths) * len(models_ids)} embeddings")


def process_img_batch(imgs_paths: list[Path], model_id: str):
    print(f"Starting processig for model {model_id}")

    model = AutoModel.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    for i, img_path in enumerate(imgs_paths):
        process_img(img_path, processor, model)
        if i % 500 == 0:
            print(model_id, f"({i} | {round(i * 100 / len(imgs_paths))}%)")

    print(f"Finished processing for  model {model_id}")


def process_img(img_path: Path, processor: AutoProcessor, model: CLIPModel):
    try:
        img_id = img_path.stem
        save_file = Path(f"data/effigy/{model.name_or_path.replace('/','-')}/{img_id}")
        if lazy and save_file.is_file():
            return

        img = Image.open(img_path)

        inputs = processor(images=img, return_tensors="pt")

        with torch.no_grad():
            img_embedding = model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )

        img_embedding_np = img_embedding.detach().numpy()[0]
        save_file.write_bytes(img_embedding_np.tobytes())

    except Exception as e:
        print(f"failed to process {img_path}: {e}")


if __name__ == "__main__":
    input_directory = Path("data/images/flowers")
    t0 = perf_counter()
    main(input_directory)
    t1 = perf_counter()
    print(f"Executed in {t1-t0}s")
