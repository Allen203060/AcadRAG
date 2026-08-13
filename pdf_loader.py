import os
import sys
import torch

# Enable TensorFloat-32 to speed up Docling's RT-DETR vision models on RTX 30-series GPUs
torch.set_float32_matmul_precision('high')

from docling.document_converter import DocumentConverter

def extract_pdf_with_docling(pdf_path: str, output_md_path: str) -> str:
    """
    Parses a digital-born academic PDF using IBM's Docling (DOM Parser + TableFormer).
    Outputs clean, layout-aware Markdown with preserved tables and headers.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        
    print(f"📄 Processing PDF with Docling: {pdf_path}...")
    
    # 1. Initialize Docling Converter
    converter = DocumentConverter()
    
    # 2. Convert PDF to Document Object Model (DOM)
    result = converter.convert(pdf_path)
    
    # 3. Export DOM tree to Markdown
    markdown_content = result.document.export_to_markdown()
    
    # 4. Save to data directory
    os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
        
    print(f"✅ Successfully converted PDF to Markdown via Docling: {output_md_path}")
    return markdown_content

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_pdf = sys.argv[1]
        out_name = os.path.splitext(os.path.basename(input_pdf))[0] + ".md"
        out_path = os.path.join("data", out_name)
        extract_pdf_with_docling(input_pdf, out_path)
    else:
        print("Usage: python pdf_loader.py <path_to_pdf>")
