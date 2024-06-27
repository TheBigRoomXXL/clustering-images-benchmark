import csv
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from random import sample

import numpy as np
from fastcluster import linkage
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster

effigies_dir = Path("data/effigy")
linkage_dir = Path("data/linkages")
graph_dir = Path("data/graphs")

image_dir = Path("data/images/flowers-clean")
image_ids = [file.stem for file in image_dir.iterdir()]

image_ids_set = set(image_ids)
effigy_id = "openai-clip-vit-base-patch32"
metric = "cosine"

n_list = [
    10,
    100,
    500,
    1_000,
    2_000,
    5_000,
    10_000,
    # 20_000,
]


def main():
    print(len(n_list), "graph to generate")

    n_max = max(n_list)
    effigies = load_data(effigy_id, n_max)

    for i, n in enumerate(n_list):
        linkage_id = f"{effigy_id}__{metric}__{n}"
        print(f"Processing {linkage_id} ({round(i*100 / len(n_list) )}%)")

        t0 = perf_counter()
        links = compute_linkage(effigies[:n], metric)
        t1 = perf_counter()

        save_linkage(linkage_id, links)

        graph = make_graph(links)
        save_graph(linkage_id, graph)

        print("  ⤷", effigy_id, metric, n, t1 - t0)


def load_data(effigy_id: str, n: int) -> NDArray:
    dtype = np.bool_
    if effigy_id.startswith("openai"):
        dtype = np.float32

    data = np.asarray(
        [
            np.fromfile(
                Path(effigies_dir).joinpath(effigy_id).joinpath(img_id), dtype=dtype
            )
            for img_id in image_ids[:n]
        ]
    )

    print("data loaded")
    return data


def compute_linkage(effigies: NDArray, m: str) -> tuple[int, NDArray]:
    links = linkage(effigies, method="ward", metric=m, preserve_input=False)
    del effigies
    return links


def save_linkage(linkage_id: str, links: NDArray):
    save_file = Path(linkage_dir.joinpath(linkage_id))
    save_file.write_bytes(links.tobytes())


def make_graph(links: NDArray):
    factors = [0.5, 0.1, 0.01]
    lmax = links[:, 2].max()
    levels = [fcluster(links, f * lmax, criterion="distance") for f in factors]
    return [
        (
            str(image_ids[i]),
            int(levels[0][i]),
            int(levels[1][i]),
            int(levels[2][i]),
        )
        for i in range(len(levels[0]))
    ]


def save_graph(linkage_id: str, graph: list[tuple]):
    save_file = Path(graph_dir.joinpath(f"{linkage_id}.json"))
    save_file.write_text(json.dumps(graph))


if __name__ == "__main__":
    t0 = perf_counter()
    main()
    t1 = perf_counter()
    print(f"Executed in {t1-t0}s")
