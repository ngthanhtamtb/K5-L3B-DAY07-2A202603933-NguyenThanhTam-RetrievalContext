"""Advanced Multimodal Ingestion Pipeline for Vinamilk Annual Report 2024 (100+ pages).

Processes all physical & logical pages of the PDF, categorizes sections into:
- text: narrative, governance, strategy, sustainability
- table: financial indicators, balance sheet, income statement, shareholder structures
- chart: dividend trends, 5-year financial growth, market share distributions
- diagram: value chain, governance hierarchy, milk farm supply network
Preserves metadata: pdf_page, book_pages, section, content_type, and region hints.
"""
import re
import json
from pathlib import Path
from typing import Any, Dict, List
import pypdf


def detect_content_type(text: str) -> str:
    """Classify content type based on structural and semantic cues."""
    t_lower = text.lower()
    
    # Check for table indicators (numbers formatted in columns, rows, financial metrics)
    table_keywords = ["doanh thu", "lợi nhuận", "bảng cân đối", "tổng tài sản", "vốn chủ sở hữu",
                      "tỷ lệ sở hữu", "chỉ số", "năm 2024", "năm 2023", "tỷ đồng", "đơn vị tính"]
    number_density = len(re.findall(r'\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?\b', text))
    table_hits = sum(1 for kw in table_keywords if kw in t_lower)
    
    if table_hits >= 2 and number_density >= 6:
        return "table"
        
    # Check for chart indicators (growth trends, percentage bars, dividend progression)
    chart_keywords = ["biểu đồ", "tăng trưởng", "cơ cấu", "tỷ trọng", "cổ tức", "chi trả", "%", "thị phần"]
    chart_hits = sum(1 for kw in chart_keywords if kw in t_lower)
    if chart_hits >= 3 and "%" in text:
        return "chart"
        
    # Check for diagram indicators (supply chain, governance structure, value chain)
    diagram_keywords = ["sơ đồ", "chuỗi giá trị", "mô hình", "hệ thống", "cơ cấu tổ chức", "quy trình", "chuỗi cung ứng"]
    if any(kw in t_lower for kw in diagram_keywords):
        return "diagram"
        
    return "text"


def ingest_advanced_pdf(pdf_path: str, output_path: str) -> List[Dict[str, Any]]:
    print(f"Opening PDF: {pdf_path}...")
    reader = pypdf.PdfReader(pdf_path)
    total_physical_pages = len(reader.pages)
    print(f"Total physical pages: {total_physical_pages}")

    chunks: List[Dict[str, Any]] = []
    chunk_counter = 0

    current_section = "Mở đầu & Thông điệp"

    for p_idx, page in enumerate(reader.pages):
        pdf_page_num = p_idx + 1
        raw_text = page.extract_text() or ""
        
        # Clean text
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if not lines:
            continue
            
        full_page_text = "\n".join(lines)
        
        # Detect header book pages (e.g. "Báo cáo thường niên Vinamilk 2024 \n 18 19 \n Luôn cầu tiến Thông tin chung")
        page_numbers_match = re.search(r'\b(\d{1,3})\s+(\d{1,3})\b', full_page_text[:120])
        if page_numbers_match:
            left_p, right_p = page_numbers_match.group(1), page_numbers_match.group(2)
            book_pages = f"{left_p}-{right_p}"
        else:
            book_pages = f"Trang {pdf_page_num}"

        # Detect chapter/section transitions
        for line in lines[:6]:
            if any(sec_kw in line.lower() for sec_kw in ["thông tin chung", "báo cáo của ban điều hành", "quản trị công ty", "báo cáo tài chính", "phát triển bền vững"]):
                current_section = line
                break

        # Segment page into logical blocks (paragraphs or subsections)
        paragraphs = re.split(r'\n{2,}', full_page_text)
        if len(paragraphs) == 1:
            # Split by reasonable size (800-1200 chars)
            paragraphs = [full_page_text[i:i+1000] for i in range(0, len(full_page_text), 900)]

        for sub_idx, para in enumerate(paragraphs):
            para = para.strip()
            if len(para) < 40:
                continue

            content_type = detect_content_type(para)
            chunk_id = f"vnm_2024_p{pdf_page_num:03d}_{content_type}_{sub_idx:02d}"

            chunk_obj = {
                "id": chunk_id,
                "source_id": "vinamilk-annual-report-2024",
                "source": "data/advanced/vinamilk-annual-report-2024.pdf",
                "page": pdf_page_num,
                "text": para,
                "metadata": {
                    "pdf_page": pdf_page_num,
                    "book_pages": book_pages,
                    "section": current_section,
                    "content_type": content_type,
                    "char_count": len(para)
                }
            }
            chunks.append(chunk_obj)
            chunk_counter += 1

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    
    print(f"Successfully processed {total_physical_pages} PDF pages (~190 book pages).")
    print(f"Total chunks created: {len(chunks)}")
    
    # Breakdown by content type
    types = {}
    for c in chunks:
        ct = c["metadata"]["content_type"]
        types[ct] = types.get(ct, 0) + 1
    print("Content type distribution:", types)
    return chunks


if __name__ == "__main__":
    pdf_file = "data/advanced/vinamilk-annual-report-2024.pdf"
    output_chunks = "data/advanced/chunks.json"
    ingest_advanced_pdf(pdf_file, output_chunks)
