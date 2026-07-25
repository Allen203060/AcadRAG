import os
import fitz  # PyMuPDF
from typing import List

def extract_pdf_to_markdown(pdf_path: str, output_md_path: str) -> str:
    """
    Extracts text from an academic PDF file and converts it into structured Markdown.
    Supports layout reading order preservation.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        
    print(f"📄 Processing PDF: {pdf_path}...")
    doc = fitz.open(pdf_path)
    markdown_content = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Extract blocks sorted by reading order (top-to-bottom, left-to-right)
        # PyMuPDF 'blocks' mode (flags=fitz.TEXT_DECOMPILE) preserves column reading order
        blocks = page.get_text("blocks", sort=True)
        
        markdown_content.append(f"\n\n# Page {page_num + 1}\n\n")
        
        for block in blocks:
            # block format: (x0, y0, x1, y1, text, block_no, block_type)
            text = block[4].strip()
            if not text:
                continue
                
            # Heuristic heading detection based on brevity & capitalization
            if len(text) < 80 and not text.endswith(".") and "\n" not in text:
                markdown_content.append(f"### {text}\n")
            else:
                markdown_content.append(f"{text}\n\n")

    full_markdown = "".join(markdown_content)
    
    # Save the output markdown into the data folder
    os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(full_markdown)
        
    print(f"✅ Successfully converted PDF to Markdown: {output_md_path}")
    return full_markdown

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        input_pdf = sys.argv[1]
        out_name = os.path.splitext(os.path.basename(input_pdf))[0] + ".md"
        out_path = os.path.join("data", out_name)
        extract_pdf_to_markdown(input_pdf, out_path)
    else:
        print("Usage: python pdf_loader.py <path_to_pdf>")
