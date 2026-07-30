import os
import re
import sys
import subprocess
import tempfile
import pymupdf as fitz

# Model Paths
MODEL_PATH = "models/unlimited_ocr/Unlimited-OCR-Q4_K_M.gguf"
MMPROJ_PATH = "models/unlimited_ocr/mmproj-Unlimited-OCR-F16.gguf" # Updated to match your download
LLAMA_CLI_PATH = "llama.cpp/build/bin/llama-cli"  # Path to compiled binary

def render_pdf_page_to_image(page, dpi: int = 200) -> str:
    """Renders a PDF page to a temporary high-DPI PNG image."""
    zoom = dpi / 72  # 72 is standard PDF points per inch
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix)
    
    temp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    pix.save(temp_img.name)
    return temp_img.name


def clean_ocr_output(raw_text: str) -> str:
    """Strips the bounding box markers from the OCR output to create clean Markdown."""
    blocks = []
    
    # Matches patterns like: text [200, 91, 800, 144]Hello World
    bbox_pattern = re.compile(r'^([a-zA-Z_]+)\s*\[\d+,\s*\d+,\s*\d+,\s*\d+\](.*)$')
    
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
            
        match = bbox_pattern.match(line)
        if match:
            category = match.group(1).strip().lower()
            content = match.group(2).strip()
            
            if category == 'title':
                blocks.append(f"## {content}")
            elif category == 'aside_text' or category == 'footer':
                blocks.append(f"_{content}_") # Format as italics
            elif category == 'equation':
                blocks.append(f"$$ {content} $$") # Format math
            else:
                blocks.append(content)
        else:
            # If it's a raw line without a bounding box, just keep it
            blocks.append(line)
            
    return "\n\n".join(blocks)


def ocr_image_with_gguf(image_path: str) -> str:
    """Executes llama-cli with Unlimited-OCR GGUF vision model to extract markdown."""
    if not os.path.exists(LLAMA_CLI_PATH):
        raise FileNotFoundError(f"llama-cli binary not found at {LLAMA_CLI_PATH}. Please compile llama.cpp first.")

    # --- ADDED <|grounding|> KEYWORD ---
    prompt = "document parsing."

    cmd = [
        LLAMA_CLI_PATH,
        "-m", MODEL_PATH,
        "--mmproj", MMPROJ_PATH,
        "--image", image_path,
        "-c", "4096",
        "-p", prompt,
        "-n", "1024",   
        "-no-cnv",  
        "-st",                 
        "--temp", "0",                      # <--- Best for strict OCR
        "-ngl", "99",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        raw_output = result.stdout.strip()
        return clean_ocr_output(raw_output)  # <--- Clean the raw output!
    except subprocess.CalledProcessError as e:
        print(f"⚠️ OCR Execution Error on image {image_path}: {e.stderr}")
        return ""

def extract_pdf_with_unlimited_ocr(pdf_path: str, output_md_path: str) -> str:
    """
    Renders academic PDF pages to images and runs Unlimited-OCR GGUF 
    to extract layout-aware Markdown.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        
    print(f"📄 Starting Unlimited-OCR GGUF Ingestion for: {pdf_path}")
    doc = fitz.open(pdf_path)
    full_markdown = []

    for page_num in range(len(doc)):
        print(f"  ──► Processing Page {page_num + 1}/{len(doc)}...")
        page = doc[page_num]
        
        # 1. Render page to image
        img_path = render_pdf_page_to_image(page, dpi=200)
        
        try:
            # 2. Run GGUF Vision OCR
            page_markdown = ocr_image_with_gguf(img_path)
            
            full_markdown.append(f"\n\n# Page {page_num + 1}\n\n")
            full_markdown.append(page_markdown)
        finally:
            # 3. Clean up temp image file
            if os.path.exists(img_path):
                os.remove(img_path)

    complete_md_content = "".join(full_markdown)

    # Save to data directory
    os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(complete_md_content)

    print(f"✅ Successfully created layout-aware Markdown: {output_md_path}")
    return complete_md_content

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_pdf = sys.argv[1]
        out_name = os.path.splitext(os.path.basename(input_pdf))[0] + ".md"
        out_path = os.path.join("data", out_name)
        extract_pdf_with_unlimited_ocr(input_pdf, out_path)
    else:
        print("Usage: python pdf_loader.py <path_to_pdf>")
