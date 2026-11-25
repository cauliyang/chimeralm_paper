"""Export Figure 1 as an editable (vector) PDF from figures/final_figures/figure1.svg.

Why this script exists
----------------------
The BioRender icons in figure1.svg are stored as a colour <image> plus a separate
<mask> holding a second <image> (the transparency), both on the same unit square
and with identical pixel sizes. Inkscape's PDF exporter rasterises SVG masks at
72 ppi, so a plain "Save as PDF" gives every icon a coarse 72-ppi outline. Here
each colour image is merged with its mask into one RGBA PNG (alpha = mask
luminance x mask alpha x image alpha) and the mask reference is dropped, so the
icon keeps its native resolution in the PDF. Text and shapes stay vector.

figure1.svg is not modified; the merged copy is written to a temporary file.

Usage: python scripts/figure1_editable_pdf.py [out.pdf]
       (default: figures/final_figures/figure1.pdf)
Needs: inkscape, pdftoppm (poppler), Pillow, numpy, pikepdf.
"""

from __future__ import annotations

import base64
import io
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pikepdf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "figures" / "final_figures" / "figure1.svg"
PAGE = 2  # page of figure1.svg that holds the Figure 1 layout
FILTER_DPI = 600  # resolution for the few elements Inkscape must rasterise (filters)
BBOX_DPI = 300  # resolution used only to find the content bounding box

NS = {
    "": "http://www.w3.org/2000/svg",
    "xlink": "http://www.w3.org/1999/xlink",
    "inkscape": "http://www.inkscape.org/namespaces/inkscape",
    "sodipodi": "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd",
    "svg": "http://www.w3.org/2000/svg",
}
SVG = "{http://www.w3.org/2000/svg}"
HREF = "{http://www.w3.org/1999/xlink}href"
DATA_URI = re.compile(r"data:image/(\w+);base64,(.*)", re.S)


def _load(href: str) -> Image.Image:
    m = DATA_URI.match(href)
    if not m:
        raise ValueError("image is not an embedded data URI")
    return Image.open(io.BytesIO(base64.b64decode(m.group(2))))


def _mask_alpha(mask: Image.Image) -> np.ndarray:
    """SVG mask value in 0..1: linear-RGB luminance of the mask content times its alpha."""
    rgba = np.asarray(mask.convert("RGBA"), dtype=np.float64) / 255.0
    srgb = rgba[..., :3]
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    lum = 0.2125 * lin[..., 0] + 0.7154 * lin[..., 1] + 0.0721 * lin[..., 2]
    return lum * rgba[..., 3]


def merge_masks(src: Path, dst: Path) -> int:
    """Fold every single-image mask into the image that uses it. Returns the number merged."""
    for prefix, uri in NS.items():
        ET.register_namespace(prefix, uri)
    tree = ET.parse(src)
    root = tree.getroot()
    masks = {m.get("id"): m for m in root.iter(SVG + "mask")}
    merged = 0
    for el in root.iter(SVG + "image"):
        ref = el.get("mask")
        if not ref:
            continue
        mask = masks[re.search(r"#([^)]+)", ref).group(1)]
        kids = list(mask)
        # Only the layout verified for this file: one <image> filling the unit square.
        if (
            len(kids) != 1
            or kids[0].tag != SVG + "image"
            or kids[0].get("transform")
            or (kids[0].get("width"), kids[0].get("height")) != (el.get("width"), el.get("height"))
            or mask.get("maskUnits") != "userSpaceOnUse"
        ):
            raise ValueError(f"mask {mask.get('id')} has an unexpected layout; refusing to guess")
        colour = _load(el.get(HREF)).convert("RGBA")
        alpha = _mask_alpha(_load(kids[0].get(HREF)))
        if alpha.shape != (colour.height, colour.width):
            raise ValueError(f"mask {mask.get('id')} and its image differ in pixel size")
        px = np.asarray(colour, dtype=np.float64)
        px[..., 3] = px[..., 3] * alpha
        buf = io.BytesIO()
        Image.fromarray(np.clip(np.rint(px), 0, 255).astype(np.uint8), "RGBA").save(buf, "PNG", optimize=True)
        el.set(HREF, "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
        del el.attrib["mask"]
        merged += 1
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    return merged


def _bbox(pdf: Path, workdir: Path) -> tuple[float, float, float, float]:
    """Bounding box (x0, y0, x1, y1 in PDF points, origin bottom-left) of the non-white content."""
    stem = workdir / "bbox"
    subprocess.run(["pdftoppm", "-r", str(BBOX_DPI), "-png", "-singlefile", str(pdf), str(stem)], check=True)
    img = np.asarray(Image.open(f"{stem}.png").convert("RGB"), dtype=np.int16)
    ink = (255 - img).max(axis=2) > 3  # anything visibly darker than white
    rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    scale = 72.0 / BBOX_DPI
    height_pt = img.shape[0] * scale
    left, right = int(cols[0]), int(cols[-1]) + 1
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    return left * scale, height_pt - bottom * scale, right * scale, height_pt - top * scale


def main() -> None:
    out = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "figures" / "final_figures" / "figure1.pdf"
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        merged_svg, raw_pdf = work / "figure1_merged.svg", work / "figure1_page.pdf"
        n = merge_masks(SRC, merged_svg)
        subprocess.run(
            ["inkscape", str(merged_svg), f"--export-page={PAGE}", "--export-area-page",
             f"--export-dpi={FILTER_DPI}", "--export-type=pdf", f"--export-filename={raw_pdf}"],
            check=True, capture_output=True,
        )
        x0, y0, x1, y1 = _bbox(raw_pdf, work)
        with pikepdf.open(raw_pdf) as pdf:
            page = pdf.pages[0]
            box = pikepdf.Array([round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)])
            page.MediaBox = box
            page.CropBox = box
            pdf.save(out)
    print(f"merged {n} icon masks; wrote {out} ({x1 - x0:.1f} x {y1 - y0:.1f} pt)")


if __name__ == "__main__":
    main()
