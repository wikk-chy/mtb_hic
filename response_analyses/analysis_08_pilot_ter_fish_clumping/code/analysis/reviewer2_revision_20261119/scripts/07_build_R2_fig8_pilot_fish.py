#!/usr/bin/env python3
"""
Build Response Fig. 8 (Reviewer 2, Comment 7 / Audit Comment 13):
Clean standalone pilot ter-probe FISH image extracted from Reviewer 4's Response Fig. 10E
so Reviewer 2 can view the cell-clumping evidence directly under Comment 7 without flipping to R4.
"""
from pathlib import Path
import fitz
from PIL import Image

ROOT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review")
PDF10 = ROOT / "Response_to_Reviewers_Overleaf/pdf_dir/Response_Fig_10.pdf"
OUT_FIG = ROOT / "analysis/reviewer2_revision_20261119/figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

doc10 = fitz.open(PDF10)
W = 595.276
# Panel E images occupy y = 445..732
H = 336.0

out = fitz.open()
page = out.new_page(width=W, height=H)
page.draw_rect(fitz.Rect(0, 0, W, H), color=None, fill=(1, 1, 1))

page.insert_text(
    (14, 18),
    "Pilot ter-probe FISH across four M. tuberculosis replicates illustrates pronounced cell clumping",
    fontname="hebo", fontsize=9.8, color=(0.12, 0.16, 0.20),
)
page.insert_text(
    (14, 31),
    "Red: ter-specific probe; blue: DAPI. Top row: overview fields (scale bar, 20 µm); bottom row: magnified insets (scale bar, 10 µm).",
    fontname="helv", fontsize=7.6, color=(0.35, 0.38, 0.42),
)

# Replicate headers above the 4 columns
col_centers = [81.1, 224.7, 368.3, 511.9]
for idx, cx in enumerate(col_centers, 1):
    lbl = f"Replicate {idx}"
    tw = fitz.get_text_length(lbl, fontname="hebo", fontsize=8.0)
    page.insert_text((cx - tw / 2, 46), lbl, fontname="hebo", fontsize=8.0, color=(0.15, 0.18, 0.22))

clip_E = fitz.Rect(0, 444.0, W, 730.0)
page.show_pdf_page(fitz.Rect(0, 49.0, W, 335.0), doc10, 0, clip=clip_E, keep_proportion=True)
doc10.close()

pdf_path = OUT_FIG / "Response_Fig_8.pdf"
png_path = OUT_FIG / "Response_Fig_8.png"
jpg_path = OUT_FIG / "Response_Fig_8.jpg"
out.save(pdf_path, garbage=4, deflate=True)
pix = out[0].get_pixmap(dpi=250, alpha=False)
pix.save(png_path)
out.close()
Image.open(png_path).convert("RGB").save(jpg_path, quality=95, dpi=(250, 250))
print("Saved Response_Fig_8.{pdf,png,jpg}")
