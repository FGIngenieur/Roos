import pdfplumber
import pandas as pd
from PIL import Image
import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_path
import tempfile
import os
from typing import (
        Literal, Optional, Tuple, Dict, List
    )

class DataLoader:
    """
    Unified file loader for CSV, XLSX, and messy PDF table extraction.
    
    Attributes:
        filepath (str): input file path
        columns (dict): column x-ranges for PDF processing
        sheet_name (str or None): optional XLSX sheet selector
        _filecsv (str): internal CSV path after conversion
    """

    def __init__(self, filepath: str, columns: dict, sheet_name: str = None, method : Literal["OCR", "Plumber", "Custom"] = "Plumber"):
        self.filepath = filepath
        self.columns = columns
        self.sheet_name = sheet_name
        self.method = method
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
    
    def _pdf_to_csv_with_table_extraction(self, pdf_path: str, output_csv: str):
        """
        Attempt to extract tables from PDF using pdfplumber's extract_table().
        Saves the result as a CSV.
        """
        all_rows = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                # Try to extract table directly
                table = page.extract_table()

                if table:
                    # append all rows from this page
                    all_rows.extend(table)
                else:
                    # If no table is found, skip page (or fallback to messy method later)
                    print(f"Page {page_number}: No table detected.")

        if not all_rows:
            raise RuntimeError("No tables detected in the PDF using extract_table().")

        # Convert to DataFrame
        df = pd.DataFrame(all_rows[1:], columns=all_rows[0])  # assume first row is header
        df.to_csv(output_csv, index=False)
        print(f"CSV saved to {output_csv}")
        return output_csv
    
    def _pdf_to_csv_with_ocr(self, pdf_path: str, output_csv: str, dpi=300):
        """
        Headless OCR table extraction for scanned/messy PDFs.
        Converts PDF pages to images, detects tables, runs OCR, and saves CSV.
        
        Requirements:
            pip install pdf2image pytesseract opencv-python-headless pandas
            sudo apt install poppler-utils  # for pdf2image on Linux
        """
        all_rows = []

        # Convert PDF pages to images
        pages = convert_from_path(pdf_path, dpi=dpi)

        for page_number, pil_image in enumerate(pages, start=1):
            # Convert PIL image to OpenCV grayscale
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Threshold to binary image
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
            )

            # Detect horizontal and vertical lines to estimate table cells
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
            table_mask = cv2.add(horizontal_lines, vertical_lines)

            # Find contours for potential cells
            contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            bounding_boxes = [cv2.boundingRect(c) for c in contours]
            bounding_boxes = sorted(bounding_boxes, key=lambda b: (b[1], b[0]))  # top->down, left->right

            page_rows = []
            for (x, y, w, h) in bounding_boxes:
                cell_img = gray[y:y+h, x:x+w]
                text = pytesseract.image_to_string(cell_img, config="--psm 7").strip()
                page_rows.append(text)

            # Group rows by approximate number of columns
            if page_rows:
                row_length = max(len(page_rows) // max(1, len(contours)), 1)
                for i in range(0, len(page_rows), row_length):
                    all_rows.append(page_rows[i:i+row_length])

        if not all_rows:
            raise RuntimeError("OCR could not detect any tables in the PDF.")

        # Save to CSV
        df = pd.DataFrame(all_rows)
        df.to_csv(output_csv, index=False)
        print(f"OCR CSV saved to {output_csv}")
        return output_csv

    # =====================================================================
    # PDF → CSV pipeline
    # =====================================================================

    def _extract_words(self, pdf_path):
        pages_words = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                pages_words.append(page.extract_words())
        return pages_words

    # =====================================================================
    # PDF column assignment (robust)
    # =====================================================================

    def _assign_columns(self, words, tolerance=2):
        """
        Assigns words to columns based on x-ranges with optional tolerance.
        Returns a list of dicts, one per line.
        """
        col_ranges = self.columns
        lines = {}

        # Group words by rounded y-coordinate (top of word)
        for w in words:
            y = round(w["top"])
            lines.setdefault(y, []).append(w)

        rows = []
        for y, ws in sorted(lines.items()):
            # Sort words left to right
            ws_sorted = sorted(ws, key=lambda w: w["x0"])
            row = {col: "" for col in col_ranges}

            for w in ws_sorted:
                x = w["x0"]
                for col, (xmin, xmax) in col_ranges.items():
                    if (xmin - tolerance) <= x <= (xmax + tolerance):
                        row[col] += " " + w["text"]
                        break

            rows.append({k: v.strip() for k, v in row.items()})
        return rows

    # =====================================================================
    # Helper: check if a string is numeric
    # =====================================================================
    def _is_numeric(self, s: str) -> bool:
        """Check if a string can be interpreted as a number (handles commas, periods, €)."""
        if not s:
            return False
        s_clean = s.replace(",", "").replace("€", "").strip()
        try:
            float(s_clean)
            return True
        except ValueError:
            return False

    # =====================================================================
    # Rebuild rows into structured items (robust)
    # =====================================================================
    def _rebuild_rows(self, raw_rows):
        """
        Merges multi-line descriptions and ensures numeric parsing.
        Returns list of dicts with Description, Quantity, Unit Price, Total.
        """
        items = []
        desc_buf = []
        qty = unit = total = ""

        for r in raw_rows:
            d = r.get("description", "").strip()
            q = r.get("qty", "").strip()
            u = r.get("unit_price", "").strip()
            t = r.get("total", "").strip()

            if d:
                desc_buf.append(d)

            if self._is_numeric(q):
                qty = q

            if self._is_numeric(u):
                unit = u

            if self._is_numeric(t):
                total = t

                merged_desc = " ".join(desc_buf)
                items.append({
                    "Description": merged_desc,
                    "Quantity": qty,
                    "Unit Price": unit,
                    "Total": total
                })

                # Reset for next row
                desc_buf = []
                qty = unit = total = ""

        # Catch remaining description if last row had no total
        if desc_buf:
            merged_desc = " ".join(desc_buf)
            items.append({
                "Description": merged_desc,
                "Quantity": qty,
                "Unit Price": unit,
                "Total": total
            })

        return items

    def _handle_pdf(self):
        """
        Entire PDF → CSV pipeline.
        Returns path to generated CSV file.
        """
        temp_csv = self.filepath + ".converted.csv"

        if self.method == "OCR":
            return self._pdf_to_csv_with_ocr(self.filepath, temp_csv)
        
        elif self.method == "Plumber":
            return self._pdf_to_csv_with_table_extraction(self.filepath, temp_csv)
        
        else:

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
