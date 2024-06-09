# Serve the web based vizualization

from pathlib import Path
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse

app = FastAPI()

app.mount("/data", StaticFiles(directory="data"), name="data")


@app.get("/")
def viz():
    print("viz")
    graph_dir = Path("data/graphs")
    graph_ids = [f for f in graph_dir.iterdir()]
    options = [f'<option value="{gid}">{gid}</option>' for gid in graph_ids]

    head = """
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="X-UA-Compatible" content="ie=edge">
        <title>Image Clustering Benchmark</title>

        <!-- Global Metadata -->
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, shrink-to-fit=no">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">

        <!-- Primary Meta Tags -->
        <title>Image Embedding Benchmark</title>
        <meta name="description" content="Image Embedding Benchmark by TheBigRoomXXL">
        <meta name="author" content="TheBigRoomXXL">

        <!-- Licence -->
        <link rel="license"
            href="https://github.com/TheBigRoomXXL/clustering-images-benchmark/main/LICENCE.md">

        <style>
            :root {
                --bg: #111;
                --fg: #ddd;
            }

            body {
                background-color: var(--bg);
                color : var(--fg);
                height: 100vh;
            }

            #sigma {
                display: flex;
                height: 100%;
            }

            s
        </style>
    </head>
    """

    body = f"""<body>
        <div>
            <label for="graph-select">Select graph</label>

            <select id="graph-select" required>
                {" ".join(options)}
            </select>
        </div>
        <div id="sigma"></div>
    </body>
    """
    script = """
        <script type="module">
            import Sigma from 'https://cdn.jsdelivr.net/npm/sigma@3.0.0-beta.19/+esm';
            import { Graph } from 'https://cdn.jsdelivr.net/npm/graphology@0.25.4/+esm';
            import { circlepack } from 'https://cdn.jsdelivr.net/npm/graphology-layout@0.6.1/+esm';
            import { NodeImageProgram } from 'https://cdn.jsdelivr.net/npm/@sigma/node-image@3.0.0-beta.9/+esm';

            const graph = new Graph();

            // Instantiate sigma.js and render the graph
            const sigmaContainer = document.getElementById("sigma")
            const sigmaInstance = new Sigma(graph, sigmaContainer , {
                allowInvalidContainer: true,
                defaultNodeType: "image",
                nodeProgramClasses: {
                    image: NodeImageProgram,
                },
                zoomToSizeRatioFunction: (x) => x,
                itemSizesReference: "positions"
            });

            updateGraph("data/graphs/openai-clip-vit-base-patch32__cosine__100.json");

            async function updateGraph(data_url) {
                const reponse = await fetch(data_url);
                const _nodes = await reponse.json();

                const nodes = _nodes.map((n) => {
                    return {
                        key: n[0],
                        attributes: {
                            x: 0,
                            y: 0,
                            l1: n[1],
                            l2: n[2],
                            l3: n[3],
                            size: 1,
                            image: `/data/images/thumbnail/${n[0]}.webp`,
                        }
                    };
                });

                const graph = new Graph();
                graph.import({ nodes });

                circlepack.assign(graph, {
                    hierarchyAttributes: ['l1', 'l2', 'l3']
                });


                sigmaInstance.setGraph(graph);
            }

            const select = document.getElementById("graph-select")
            select.addEventListener("change", ()=> {
                updateGraph(select.value)
            }) 

        </script>
    """

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
        {head}
        {body}
        {script}
    </html>
    """

    return Response(html, media_type="text/html")
