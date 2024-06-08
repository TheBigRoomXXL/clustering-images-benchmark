import csv
import json
from pathlib import Path
from datetime import datetime
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from fastcluster import linkage
from scipy.cluster.hierarchy import fcluster

linkage_dir = Path("data/linkages")
graph_dir = Path("data/graphs")
effigies_dir = Path("data/effigy")
image_dir = Path("data/images/128")
image_ids = [file.stem for file in image_dir.iterdir()]
effigies_ids = [
    "average_hash-08",
    "average_hash-16",
    "average_hash-32",
    # "average_hash-64",
    "phash-08",
    "phash-16",
    "phash-32",
    # "phash-64",
    "dhash-08",
    "dhash-16",
    "dhash-32",
    # "dhash-64",
    "colorhash-08",
    "colorhash-16",
    "colorhash-32",
    # "colorhash-64",
    # "openai-clip-vit-base-patch16",
    # "openai-clip-vit-base-patch32",
]
effigies_ids.sort()
metric_list = [
    # "euclidean",
    "cosine",
    "hamming",
]
n_list = [
    10,
    100,
    500,
    1000,
    2000,
    5000,
    10000,
]


def main():
    results = []
    i = 0
    nb_test = len(effigies_ids) * len(metric_list) * len(n_list)
    print(nb_test, "tests to run")
    for effigy_id in effigies_ids:
        for m in metric_list:
            for n in n_list:
                linkage_id = f"{effigy_id}__{m}__{n}"
                print(f"Processing {linkage_id} ({round(i*100 / nb_test )}%)")
                i += 1

                effigies = load_data(effigy_id, n)
                size = effigies[0].size

                t0 = perf_counter()
                links = compute_linkage(effigies, m)
                t1 = perf_counter()

                save_linkage(linkage_id, links)

                graph = make_graph(links)
                save_graph(linkage_id, graph)

                results.append((effigy_id, size, m, n, t1 - t0))
                print("  ⤷", results[-1])

    report = Path(f"data/reports/{round(datetime.now().timestamp())}.csv")
    with report.open("w") as f:
        writer = csv.writer(f)
        writer.writerows(results)


def load_data(effigy_id: str, n: int) -> list[NDArray]:
    directory = effigies_dir.joinpath(effigy_id)

    dtype = np.bool_
    if effigy_id.startswith("openai"):
        dtype = np.float32
    data = [np.fromfile(path, dtype=dtype) for path in directory.iterdir()]
    return data[:n]


def compute_linkage(effigies: list[NDArray], m: str) -> tuple[int, NDArray]:
    links = linkage(effigies, method="ward", metric=m, preserve_input=False)
    del effigies
    return links


def save_linkage(linkage_id: str, links: NDArray):
    save_file = Path(linkage_dir.joinpath(linkage_id))
    save_file.write_bytes(links.tobytes())


def make_graph(links: NDArray):
    factors = [0.5, 0.1, 0.01]
    max_dist = np.max(links[:, 2])
    levels = [fcluster(links, f * max_dist, criterion="distance") for f in factors]
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
