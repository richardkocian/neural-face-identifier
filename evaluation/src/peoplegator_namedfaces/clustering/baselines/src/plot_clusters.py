"""Create montages of faces per cluster from JSONL mapping output.

Reads a JSON Lines file produced by `cluster_embeddings.py` where each line
is an object with keys: "face", "cluster", "cluster_score". The script
groups entries by cluster and writes a PNG montage for each cluster.

Usage examples:
  python -m plot_clusters \
    --mapping clusters.jsonl --images-dir ./faces --out-dir cluster_plots --examples 16
"""
import argparse
from collections.abc import Iterable
from itertools import groupby
import json
import math
import hashlib
import colorsys
from collections import defaultdict
from os import path
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from peoplegator_namedfaces.clustering.evaluation.src.schemas import BaseModel, PeopleGatorNamedFaces__NamedFaceGroundTruth, PeopleGatorNamedFaces__ClusterPrediction
from typing import TypeVar

T = TypeVar("T", bound=BaseModel)

def read_jsonl(path: Path, type: type[T]) -> list[T]:
    items: list[T] = []
    with path.open("r", encoding="utf8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(type(**json.loads(line)))
            except Exception:
                continue
    return items

def load_data(clusters_path: Path, ground_truth_path: Path) -> tuple[list[PeopleGatorNamedFaces__ClusterPrediction], list[PeopleGatorNamedFaces__NamedFaceGroundTruth]]:
    if not clusters_path.exists():
        raise FileNotFoundError(f"Clusters mapping file not found: {clusters_path}")
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")
    return read_jsonl(clusters_path, PeopleGatorNamedFaces__ClusterPrediction), read_jsonl(ground_truth_path, PeopleGatorNamedFaces__NamedFaceGroundTruth)

def _name_to_color(name: str) -> tuple[int, int, int]:
    """Deterministically map a name to a distinct RGB color."""
    if name == "__background__":
        return (10, 10, 10)  # fixed grey for background
    # Use a hash of the name to get a consistent color
    h = hashlib.sha256(name.encode("utf8")).hexdigest()
    # Take first 6 chars for RGB, convert from hex to int
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    # Optionally adjust saturation/brightness for better visibility
    hls = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    # Make colors noticeably darker and slightly more saturated for visibility
    # reduce lightness substantially and clamp to a safe minimum
    new_lightness = 0.3
    new_saturation = 1
    new_rgb = colorsys.hls_to_rgb(hls[0], new_lightness, new_saturation)
    return (int(new_rgb[0] * 255), int(new_rgb[1] * 255), int(new_rgb[2] * 255))


def _get_text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont | ImageFont.FreeTypeFont | None) -> tuple[float, float]:
    """Return (width, height) for rendered text using available Pillow APIs.

    Tries `font.getsize`, then `draw.textbbox`, then falls back to a conservative estimate.
    """
    try:
        if font is not None and hasattr(font, "getsize"):
            textsize = font.getsize(text)
            return (float(textsize[0]), float(textsize[1]))
    except Exception:
        pass
    try:
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), text, font=font)
            return (bbox[2] - bbox[0], bbox[3] - bbox[1])
    except Exception:
        pass
    # last resort: approximate width by chars
    return (max(10, len(text) * 6), 10)


def cluster_image(data_root, face_to_gt, faces: Iterable[PeopleGatorNamedFaces__ClusterPrediction], examples: int, thumb_size: tuple[int, int] = (128, 128)) -> Image.Image:
    items: list[tuple[Path, str]] = []
    
    for item in faces:
        gt = face_to_gt.get(item.face)
        if gt is None:
            print(
                f"Warning: no ground truth found for face {item.face}, using background")
            name = "__background__"
        else:
            name = gt.person_name
        p = data_root / item.face
        items.append((p, name))
    items = items[:examples]
    n = len(items)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    tw, th = thumb_size
    i = 0
    montage = Image.new("RGB", (cols * tw, rows * th), (245, 245, 245))
    # font
    try:
        font = ImageFont.truetype("Ubuntu Monospace", 32)
    except Exception:
        font = None
        
    for r in range(rows):
        for c in range(cols):
            if i >= n:
                break
            p, name = items[i]
            cell = Image.new("RGB", (tw, th), (230, 230, 230))
            draw = ImageDraw.Draw(cell)
            try:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    im.thumbnail((tw - 8, th - 24), Image.LANCZOS)
                    x = (tw - im.width) // 2
                    y = (th - im.height) // 2
                    # leave room at bottom for text
                    if y + im.height > th - 18:
                        y = max(0, th - 18 - im.height)
                    cell.paste(im, (x, y))
            except Exception:
                # leave blank cell on error
                pass
            color = _name_to_color(name)
            # draw border (thickness 3)
            thickness = max(2, tw // 64)
            for t in range(thickness):
                draw.rectangle([t, t, tw - 1 - t, th - 1 - t], outline=color)
            text = str(name)
            # fallback font handling
            f = font
            w, htxt = _get_text_size(draw, text, f)
            tx = (tw - w) // 2
            ty = th - htxt - 7
            # draw with light stroke for readability
            try:
                draw.text((tx, ty), text, fill=color, font=f,
                            stroke_width=1, stroke_fill=(255, 255, 255))
            except TypeError:
                # older Pillow may not support stroke_width
                # draw shadow then text
                draw.text((tx + 1, ty + 1), text,
                            fill=(255, 255, 255), font=f)
                draw.text((tx, ty), text, fill=color, font=f)
            montage.paste(cell, (c * tw, r * th))
            i += 1
    return montage
    
    
    

def create_montages(clusters: list[PeopleGatorNamedFaces__ClusterPrediction], ground_truths: list[PeopleGatorNamedFaces__NamedFaceGroundTruth], data_root: Path, cluster_output: Path, examples: int, thumb_size: tuple[int, int] = (128, 128)) -> None:
    """Create montage from list of (image_path, gt_name).

    Draw colored border and write the `gt_name` in the same color.
    Missing names use grey.
    """
    face_to_gt = {gt.face: gt for gt in ground_truths}
    cluster_output.mkdir(parents=True, exist_ok=True)
    for cluster, faces in groupby(sorted(clusters, key=lambda x: x.cluster), key=lambda x: x.cluster):
        image = cluster_image(data_root, face_to_gt, faces, examples, thumb_size)
        image.save(cluster_output / f"cluster_{cluster}.png")


def main(argv=None):
    p = argparse.ArgumentParser(description="Plot cluster montages from JSONL mapping")
    p.add_argument("--clusters", "-c", required=True, help="JSON Lines mapping file (face, cluster, cluster_score)")
    p.add_argument("--ground-truth", "-g", required=True, help="JSON Lines ground truth file (face, person_name)")
    p.add_argument("--image-root", "-i", help="Base directory to resolve relative image paths")
    p.add_argument("--output", "-o", default="cluster_plots", help="Output directory for montages")
    p.add_argument("--thumb-size", "-t", type=int, default=128, help="Thumbnail max side in pixels (square)")
    p.add_argument("--examples", "-e", type=int, default=16, help="Number of examples to show per cluster")
    args = p.parse_args(argv)

    clusters, ground_truths = load_data(Path(args.clusters), Path(args.ground_truth))
    data_root = Path(args.image_root) if args.image_root else Path(".")
    create_montages(clusters, ground_truths, data_root, Path(args.output), args.examples, thumb_size=(args.thumb_size, args.thumb_size))

if __name__ == "__main__":
    main()
