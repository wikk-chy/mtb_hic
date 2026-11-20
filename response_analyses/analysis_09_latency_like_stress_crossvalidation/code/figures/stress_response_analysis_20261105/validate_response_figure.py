from pathlib import Path
import pypdfium2 as pdfium
import xml.etree.ElementTree as ET

out = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/figures/stress_response_analysis_20261105")
doc = pdfium.PdfDocument(out / "Response_Fig12_Persistence_confounders.pdf")
assert len(doc) == 1
page = doc[0]
width, height = page.get_size()
text = page.get_textpage().get_text_range()
for title in ["No coordinated SOS activation", "Substantial shared transcriptional response",
              "Five of six global structural metrics align", "Limited bin-level structural concordance",
              "863/896", "Spearman ρ = 0.250"]:
    assert title in text, title
root = ET.parse(out / "Response_Fig12_Persistence_confounders.svg").getroot()
text_elements = sum(element.tag.endswith("text") for element in root.iter())
assert text_elements > 80
print(f"PDF size: {width/72:.2f} x {height/72:.2f} in")
print(f"PDF extractable characters: {len(text)}")
print(f"SVG editable text elements: {text_elements}")
print("All panel titles and key statistics verified")
