from pathlib import Path
import pypdfium2 as pdfium
import xml.etree.ElementTree as ET

out = Path(__file__).resolve().parents[1] / "results"
reader = pdfium.PdfDocument(out / "Response_Fig4_NapM_dependence.pdf")
assert len(reader) == 1
page = reader[0]
width_pt, height_pt = page.get_size()
width_in = width_pt / 72
height_in = height_pt / 72
text = page.get_textpage().get_text_range()
assert abs(width_in - 8.27) < 0.01
assert all(label in text for label in ["Condition-dependent napM expression", "Genome-wide genotype-dependent structural response", "Independent samples confirm altered response magnitude", "NapM affects response magnitude more than direction", "Hypoxia response overlaps a pre-existing NapM-associated state", "Latency shows a related but stronger WT signature"])
root = ET.parse(out / "Response_Fig4_NapM_dependence.svg").getroot()
svg_text = sum(1 for element in root.iter() if element.tag.endswith("text"))
assert svg_text >= 100
print(f"PDF size: {width_in:.2f} x {height_in:.2f} in")
print(f"PDF extractable text: {len(text)} characters")
print(f"SVG editable text elements: {svg_text}")
print("All six panel titles verified")
