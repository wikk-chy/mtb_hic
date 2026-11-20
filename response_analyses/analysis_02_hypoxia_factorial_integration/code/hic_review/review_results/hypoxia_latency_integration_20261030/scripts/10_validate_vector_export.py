#!/usr/bin/env python3
"""Verify editable text and genuine vector export for the complete figure."""
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
import pypdfium2 as pdfium

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
OUT = ROOT / 'hic_review/review_results/hypoxia_latency_integration_20261030'
FIG, TAB = OUT / 'figures', OUT / 'tables'
pdf = FIG / 'NapM_hypoxia_full_mechanism_evidence.pdf'
svg = FIG / 'NapM_hypoxia_full_mechanism_evidence.svg'
checks = []


def check(name, condition, detail=''):
    checks.append({'check': name, 'passed': bool(condition), 'detail': detail})


doc = pdfium.PdfDocument(str(pdf))
check('vector PDF is one page', len(doc) == 1, f'pages={len(doc)}')
page = doc[0]
text_page = page.get_textpage()
text = text_page.get_text_range()
check('PDF contains substantial extractable text', len(text) > 1500, f'characters={len(text)}')
for label in ['Genotype', '730 interaction DEGs', 'Descriptive response-shape clusters',
              'DosR/DosS region', 'SigE region', 'oriC/dnaA region', 'ESX-2 region',
              'Normalized log2CPM', 'NapM ChIP']:
    check(f'PDF editable text contains: {label}', label in text)
doc.close()

root = ET.parse(svg).getroot()
ns = {'svg': 'http://www.w3.org/2000/svg'}
text_nodes = root.findall('.//svg:text', ns)
path_nodes = root.findall('.//svg:path', ns)
image_nodes = root.findall('.//svg:image', ns)
check('SVG contains editable text elements', len(text_nodes) > 150, f'text_elements={len(text_nodes)}')
check('SVG contains vector path elements', len(path_nodes) > 1000, f'path_elements={len(path_nodes)}')
check('SVG uses embedded images only for data panels', 0 < len(image_nodes) < 100,
      f'image_elements={len(image_nodes)}')
svg_text = ''.join(''.join(node.itertext()) for node in text_nodes)
for label in ['730 interaction DEGs', 'DosR/DosS region', 'SigE region', 'oriC/dnaA region', 'ESX-2 region']:
    check(f'SVG text element contains: {label}', label in svg_text)

script = (OUT / 'scripts/08_build_full_mechanism_figure.py').read_text()
check('PDF is saved directly by Matplotlib', "fig.savefig(pdf, format='pdf'" in script)
check('SVG text is preserved rather than converted to paths', "'svg.fonttype': 'none'" in script)
check('no whole-page raster PDF assembly', 'drawImage' not in script and 'reportlab' not in script)
check('scatter points are not rasterized', 'rasterized=True' not in script)
check('vector deliverables are nonempty', pdf.stat().st_size > 100000 and svg.stat().st_size > 100000,
      f'PDF={pdf.stat().st_size}; SVG={svg.stat().st_size}')

result = pd.DataFrame(checks)
result.to_csv(TAB / 'Vector_export_validation.csv', index=False)
print(result.to_string(index=False))
print(f'\n{int(result.passed.sum())}/{len(result)} checks passed')
if not result.passed.all():
    raise SystemExit('Vector export validation failed')
