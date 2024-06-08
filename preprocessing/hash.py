from pathlib import Path
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image
from imagehash import average_hash, phash, dhash, colorhash


nb_workers = 3
input_directory = "data/images/128"
lazy = True

def main():
    directory = Path(input_directory)
    imgs = directory.iterdir()

    executor = ProcessPoolExecutor(max_workers=nb_workers)
    futures = [executor.submit(process_img, ipath) for ipath in imgs]

    i = 0
    for f in as_completed(futures):
        i += 1
        if i % 1000 == 0:
            print(
                f"processed {i} out of {len(futures)} images ({round(i*100/len(futures))}%)"
            )

    print({len(futures) * 4 * 4}, "hash computed")


def process_img(img_path: Path):
    try:
        img_id = img_path.stem
        img = Image.open(img_path, formats=("WEBP",))

        for n in (8, 16, 32, 64):
            for hashfunc in (average_hash, phash, dhash, colorhash):
                save_file = Path(f"data/effigy/{hashfunc.__name__}-{n:02}/{img_id}")
                if lazy and save_file.is_file():
                    continue
                save_file.write_bytes(hashfunc(img, n).hash.tobytes())

    except Exception as e:
        print(f"failed to process {img_path}: {e}")


if __name__ == "__main__":
    t0 = perf_counter()
    main()
    t1 = perf_counter()
    print(f"Executed in {t1-t0}s")
