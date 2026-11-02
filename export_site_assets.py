"""Copy the site's documents/images locally; no runtime dependency on the parent repo."""
from pathlib import Path
import shutil


def export(root=None):
    root = Path(root) if root else Path(__file__).resolve().parent
    destination = root / "site/assets"
    destination.mkdir(parents=True, exist_ok=True)
    sources = ["Poster_YoloV1.png", "yolo.pdf", "figures/mean_loss_plot.png",
               "figures/train_map_plot.png", "docs/LITERATURE.md",
               "docs/EXPERIMENT_PROTOCOL.md", "docs/REPRODUCIBILITY.md"]
    for name in sources:
        source = root / name
        shutil.copyfile(source, destination / source.name)
    # Markdown links are made relative to the flattened standalone asset folder.
    readme = (root / "README.md").read_text(encoding="utf-8")
    readme = readme.replace("](docs/", "](").replace("](figures/", "](")
    readme = readme.replace("](site/index.html)", "](../index.html)")
    (destination / "README.md").write_text(readme, encoding="utf-8")
    print(f"Exported {len(sources) + 1} local site assets to {destination}")


if __name__ == "__main__":
    export()
