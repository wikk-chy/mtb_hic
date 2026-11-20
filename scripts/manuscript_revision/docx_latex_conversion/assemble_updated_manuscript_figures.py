#!/usr/bin/env python3
"""Back up, assemble and export the actual Revised manuscript figures."""
from pathlib import Path
import shutil
import json
import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "Establishment_and_remodeling_of_the_Mycobacterium_tuberculosis"
RESULTS = ROOT / "hic_review/review_results"
WORK = ROOT / "tmp/manuscript_update"
BACKUP = PROJECT / "backups/20261112_cid_fig_update"
BACKUP.mkdir(parents=True, exist_ok=True)

for source in [PROJECT / "main.tex", PROJECT / "figures/Fig-1-updated.jpg",
               PROJECT / "figures/Fig-2-updated.jpg", RESULTS / "Fig-1.pdf", RESULTS / "Fig-2.pdf"]:
    target = BACKUP / source.name
    if not target.exists():
        shutil.copy2(source, target)

with pymupdf.open(RESULTS / "Fig-1.pdf") as document:
    page = document[0]
    for name, clip in {"Fig1d_current": pymupdf.Rect(12, 323, 315, 468),
                       "Fig1f_current": pymupdf.Rect(462, 145, 595, 315)}.items():
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(3, 3), clip=clip, alpha=False)
        pixmap.save(WORK / f"{name}.png")

source = pymupdf.open(BACKUP / "Fig-2.pdf")
composite = pymupdf.open()
page = composite.new_page(width=source[0].rect.width, height=source[0].rect.height)
page.show_pdf_page(page.rect, source, 0)
page.add_redact_annot(pymupdf.Rect(0, 0, 247, 294), fill=(1, 1, 1))
page.add_redact_annot(pymupdf.Rect(247, 0, 347, 294), fill=(1, 1, 1))
page.apply_redactions()
with pymupdf.open(WORK / "Fig2a_final_size.pdf") as panel:
    page.show_pdf_page(pymupdf.Rect(0, 0, 247, 250), panel, 0, keep_proportion=False)
with pymupdf.open(WORK / "Fig2b_final_size.pdf") as panel:
    page.show_pdf_page(pymupdf.Rect(247, 0, 347, 112), panel, 0, keep_proportion=False)
new_fig2 = WORK / "Fig-2-final.pdf"
composite.save(new_fig2, garbage=4, deflate=True)
composite.close()
source.close()
shutil.copy2(new_fig2, RESULTS / "Fig-2.pdf")


def export_jpg(pdf_path, jpg_path):
    with pymupdf.open(pdf_path) as document:
        pixmap = document[0].get_pixmap(matrix=pymupdf.Matrix(400 / 72, 400 / 72), alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        image.save(jpg_path, quality=98, subsampling=0, dpi=(400, 400), optimize=True)
        return list(image.size)


jpg1_size = export_jpg(RESULTS / "Fig-1.pdf", PROJECT / "figures/Fig-1-updated.jpg")
jpg2_size = export_jpg(RESULTS / "Fig-2.pdf", PROJECT / "figures/Fig-2-updated.jpg")
report = {
    "backup_directory": str(BACKUP),
    "figure1_pdf": str(RESULTS / "Fig-1.pdf"),
    "figure1_action": "Reused current vector PDF containing updated CID-boundary and gene-set co-expression panels",
    "figure2_pdf": str(RESULTS / "Fig-2.pdf"),
    "figure2_action": "Replaced panels a and b at final physical sizes; all other panels preserved",
    "figure1_jpg_dimensions": jpg1_size,
    "figure2_jpg_dimensions": jpg2_size,
    "cid_boundaries": 18,
    "loops": 41,
    "loop_anchor_bins": 75,
}
(WORK / "figure_update_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
