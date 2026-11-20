#!/usr/bin/env python3
"""Render the final Figure 1/2 PDFs as high-quality, white-background JPGs."""

import json
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE_DIR = WORKSPACE / "hic_review" / "review_results"
OUTPUT_DIR = (
    WORKSPACE
    / "Establishment_and_remodeling_of_the_Mycobacterium_tuberculosis"
    / "figures"
)
PREVIEW_DIR = WORKSPACE / "converted_previews"
DPI = 400

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

manifest = []
for stem in ("Fig-1", "Fig-2"):
    source = SOURCE_DIR / f"{stem}.pdf"
    destination = OUTPUT_DIR / f"{stem}-updated.jpg"
    preview_path = PREVIEW_DIR / f"{stem}-updated-preview.jpg"

    document = pdfium.PdfDocument(str(source))
    if len(document) != 1:
        raise ValueError(f"Expected a one-page figure: {source}")
    page = document[0]
    width_pt, height_pt = page.get_size()
    image = page.render(scale=DPI / 72).to_pil()

    if image.mode in ("RGBA", "LA"):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.getchannel("A"))
        image = background
    else:
        image = image.convert("RGB")

    image.save(
        destination,
        format="JPEG",
        quality=98,
        subsampling=0,
        optimize=True,
        dpi=(DPI, DPI),
    )

    preview = image.copy()
    preview.thumbnail((1600, 2200), Image.Resampling.LANCZOS)
    preview.save(preview_path, format="JPEG", quality=90, optimize=True)

    expected_width = width_pt / 72 * DPI
    expected_height = height_pt / 72 * DPI
    manifest.append(
        {
            "source_pdf": str(source),
            "output_jpg": str(destination),
            "preview_jpg": str(preview_path),
            "dpi": DPI,
            "page_points": [width_pt, height_pt],
            "pixels": list(image.size),
            "expected_pixels_rounded": [round(expected_width), round(expected_height)],
            "aspect_ratio_source": width_pt / height_pt,
            "aspect_ratio_output": image.width / image.height,
            "jpg_bytes": destination.stat().st_size,
            "preview_bytes": preview_path.stat().st_size,
        }
    )
    document.close()

manifest_path = PREVIEW_DIR / "conversion_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
