#!/usr/bin/env python3
"""
Build Response Fig. 5 (Reviewer 2, Comment 4 / Audit Comment 10):
Self-contained composite figure directly placed after Reviewer 2 Comment 4,
combining vector panels from Reviewer 4's stress-remodeling (Response Fig. 10A-D)
and polymer-modeling (Response Fig. 11A, D, E, F) analyses with clean sequential
labels A-G (fixing the missing-H jump in original Fig. 11).
"""
from pathlib import Path
import fitz
from PIL import Image

ROOT = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review")
PDF10 = ROOT / "Response_to_Reviewers_Overleaf/pdf_dir/Response_Fig_10.pdf"
PDF11 = ROOT / "Response_to_Reviewers_Overleaf/pdf_dir/Response_Fig_11.pdf"
OUT_FIG = ROOT / "analysis/reviewer2_revision_20261119/figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

W = 595.276
# Layout:
#   Top title bar:         y = 0..24
#   Block 1 (Fig10 A-D):   y = 24..444  (height 420)
#   Block 2 (Fig11 A -> E):y = 450..574 (height 124)
#   Block 3 (left: Fig11 D+E -> F; right: Fig11 F -> G): y = 582..742 (height 160)
H = 754.0

out = fitz.open()
page = out.new_page(width=W, height=H)
page.draw_rect(fitz.Rect(0, 0, W, H), color=None, fill=(1, 1, 1))

# Clean main title
page.insert_text(
    (16, 17),
    "Spatially non-uniform stress remodelling and polymer-model evaluation",
    fontname="hebo", fontsize=10.5, color=(0.12, 0.16, 0.20),
)

doc10 = fitz.open(PDF10)
doc11 = fitz.open(PDF11)

# 1. Panels A-D from Fig 10 (y=0..420)
clip_10 = fitz.Rect(0, 0, 595.276, 420.0)
page.show_pdf_page(fitz.Rect(0, 24.0, W, 444.0), doc10, 0, clip=clip_10, keep_proportion=True)

# Subtle horizontal divider separating Hi-C data (A-D) from polymer working models (E-G)
page.draw_line((16, 447.0), (W - 16, 447.0), color=(0.82, 0.85, 0.88), width=0.6)

# 2. Panel E: 5 simulated contact maps (from Fig 11 Panel A, y=18..142)
clip_11A = fitz.Rect(0, 18.0, 595.276, 142.0)
dest_E = fitz.Rect(0, 452.0, W, 576.0)
page.show_pdf_page(dest_E, doc11, 0, clip=clip_11A, keep_proportion=True)
# Cover old "A" label and stamp clean "E"
page.draw_rect(fitz.Rect(8, 454, 28, 474), color=None, fill=(1, 1, 1))
page.insert_text((13, 467), "E", fontname="hebo", fontsize=11.0, color=(0, 0, 0))

# 3. Panel F (left): held-out ori-ter O/E and arm-arm O/E (from Fig 11 Panels D & E, x=336..592, y=151..286)
clip_11DE = fitz.Rect(336.0, 151.0, 592.0, 286.0)
dest_F = fitz.Rect(28.0, 592.0, 308.0, 740.0)
page.show_pdf_page(dest_F, doc11, 0, clip=clip_11DE, keep_proportion=True)
# Cleanly mask original sub-labels D and E inside the clipped vector region
page.draw_rect(fitz.Rect(28, 590, 44, 606), color=None, fill=(1, 1, 1))
page.draw_rect(fitz.Rect(160, 590, 176, 606), color=None, fill=(1, 1, 1))
page.insert_text((13, 598), "F", fontname="hebo", fontsize=11.0, color=(0, 0, 0))

# 4. Panel G (right): AIC model comparison (from Fig 11 Panel F, x=8..282, y=296..518)
clip_11F = fitz.Rect(8.0, 296.0, 282.0, 518.0)
dest_G = fitz.Rect(342.0, 584.0, 576.0, 744.0)
page.show_pdf_page(dest_G, doc11, 0, clip=clip_11F, keep_proportion=True)
# Cover old "F" label if clipped inside top-left
page.draw_rect(fitz.Rect(340, 584, 368, 605), color=None, fill=(1, 1, 1))
page.insert_text((328, 598), "G", fontname="hebo", fontsize=11.0, color=(0, 0, 0))

doc10.close()
doc11.close()

pdf_path = OUT_FIG / "Response_Fig_5.pdf"
png_path = OUT_FIG / "Response_Fig_5.png"
jpg_path = OUT_FIG / "Response_Fig_5.jpg"
out.save(pdf_path, garbage=4, deflate=True)
pix = out[0].get_pixmap(dpi=250, alpha=False)
pix.save(png_path)
out.close()
Image.open(png_path).convert("RGB").save(jpg_path, quality=95, dpi=(250, 250))
print("Saved Response_Fig_5.{pdf,png,jpg}")
