import os
cwd = os.getcwd()
os.chdir('../libs/')
from quotesLibs import *
os.chdir(cwd)

class DataLoader:
    """
    Unified file loader for CSV, XLSX, and messy PDF table extraction.
    
    Attributes:
        filepath (str): input file path
        columns (dict): column x-ranges for PDF processing
        sheet_name (str or None): optional XLSX sheet selector
        _filecsv (str): internal CSV path after conversion
    """

    def __init__(self, filepath: str, columns: dict, sheet_name: str = None):
        self.filepath = filepath
        self.columns = columns
        self.sheet_name = sheet_name
        self._filecsv = None

        self._initialize_csv()

    # =====================================================================
    # Internal dispatcher
    # =====================================================================

    def _initialize_csv(self):
        ext = os.path.splitext(self.filepath)[1].lower()

        if ext == ".pdf":
            self._filecsv = self._handle_pdf()

        elif ext == ".xlsx":
            self._filecsv = self._handle_xlsx()

        elif ext == ".csv":
            self._filecsv = self.filepath

        else:
            raise ValueError(f"Unsupported file type: {ext}")

    # =====================================================================
    # PDF → CSV pipeline
    # =====================================================================

    def _extract_words(self, pdf_path):
        pages_words = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                pages_words.append(page.extract_words())
        return pages_words

    def _assign_columns(self, words):
        col_ranges = self.columns
        lines = {}

        for w in words:
            y = round(w["top"])
            lines.setdefault(y, []).append(w)

        rows = []
        for y, ws in sorted(lines.items()):
            row = {col: "" for col in col_ranges}
            for w in ws:
                x = w["x0"]
                for col, (xmin, xmax) in col_ranges.items():
                    if xmin <= x < xmax:
                        row[col] += " " + w["text"]
                        break
            rows.append({k: v.strip() for k, v in row.items()})
        return rows

    def _rebuild_rows(self, raw_rows):
        items = []
        desc_buf = []
        qty = unit = total = ""

        for r in raw_rows:
            d = r.get("description", "")
            q = r.get("qty", "")
            u = r.get("unit_price", "")
            t = r.get("total", "")

            if d:
                desc_buf.append(d)

            if q.isdigit():
                qty = q

            if u.replace(",", "").replace(".", "").isdigit() or "€" in u:
                unit = u

            if t.replace(",", "").replace(".", "").isdigit() or "€" in t:
                total = t

                merged_desc = " ".join(desc_buf)
                items.append({
                    "Description": merged_desc,
                    "Quantity": qty,
                    "Unit Price": unit,
                    "Total": total
                })

                # reset
                desc_buf = []
                qty = unit = total = ""

        return items

    def _handle_pdf(self):
        """
        Entire PDF → CSV pipeline.
        Returns path to generated CSV file.
        """
        temp_csv = self.filepath + ".converted.csv"

        all_pages = self._extract_words(self.filepath)

        raw = []
        for words in all_pages:
            raw.extend(self._assign_columns(words))

        items = self._rebuild_rows(raw)

        df = pd.DataFrame(items)
        df.to_csv(temp_csv, index=False)

        return temp_csv

    # =====================================================================
    # XLSX → CSV
    # =====================================================================

    def _handle_xlsx(self):
        temp_csv = self.filepath + ".converted.csv"

        try:
            if self.sheet_name is None:
                df = pd.read_excel(self.filepath)
            else:
                df = pd.read_excel(self.filepath, sheet_name=self.sheet_name)

            df.to_csv(temp_csv, index=False)
            return temp_csv

        except Exception as e:
            raise RuntimeError(f"Failed to load XLSX file: {e}")

    # =====================================================================
    # PUBLIC METHODS
    # =====================================================================

    def saveToCSV(self, filename: str):
        """
        Saves the internally loaded CSV to a new filename.
        """
        df = self.getPandas()
        df.to_csv(filename, index=False)

    def getPandas(self) -> pd.DataFrame:
        """
        Returns the internally loaded CSV as a pandas DataFrame.
        """
        if self._filecsv is None:
            raise RuntimeError("CSV file not initialized.")
        return pd.read_csv(self._filecsv)
