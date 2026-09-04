import os
import sys
import warnings

# Suppress verbose PyTorch Dynamo JIT compiler warnings & Transformers deprecation notices
os.environ["TORCH_LOGS"] = "-dynamo"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import torch

# Enable TensorFloat-32 to speed up Docling's RT-DETR vision models on RTX 30-series GPUs
torch.set_float32_matmul_precision('high')

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions

def extract_pdf_with_docling(pdf_path: str, output_md_path: str) -> str:
    """
    Parses a digital-born academic PDF using IBM's Docling (DOM Parser + TableFormer).
    Outputs clean, layout-aware Markdown with preserved tables and headers.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        
    print(f"📄 Processing PDF with Docling: {pdf_path}...")
    
    # 1. Initialize Pipeline Options for Extreme Speed
    # Check if GPU has at least 1GB of free VRAM; otherwise fallback to CPU to avoid competing with Ollama
    free_vram_bytes = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
    device = "cuda" if free_vram_bytes > (1024 * 1024 * 1024) else "cpu"
    
    print(f"⚙️ Docling using device: {device.upper()} (Free VRAM: {round(free_vram_bytes / (1024**2), 2)} MB)")

    pipeline_options = PdfPipelineOptions(do_ocr=False)
    pipeline_options.accelerator_options = AcceleratorOptions(num_threads=8, device="cuda")

    # 2. Initialize Docling Converter with the fast options
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    # 3. Convert PDF to Document Object Model (DOM)
    result = converter.convert(pdf_path)
    
    # 4. Export DOM tree to Markdown
    markdown_content = result.document.export_to_markdown()
    
    # 5. Save to data directory
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
        print("Usage: python -m src.ingestion.pdf_loader <path_to_pdf>")
