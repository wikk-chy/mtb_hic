from pathlib import Path
import pandas as pd

base = Path("/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/analysis/ppi_source_check")
for filename in ["pr100808n_si_009.xls", "pr100808n_si_011.xls"]:
    path = base / filename
    print(f"\n=== {filename} ===")
    book = pd.ExcelFile(path)
    print("Sheets:", book.sheet_names)
    for sheet in book.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet, header=None)
        print(f"[{sheet}] shape={frame.shape}")
        print(frame.iloc[:5, :6].to_string(index=False, header=False))
        mask = frame.apply(lambda column: column.astype(str).str.contains("Rv0047c", case=False, regex=False)).any(axis=1)
        if mask.any():
            print("Rv0047c rows:")
            print(frame.loc[mask].to_string(index=False, header=False))
