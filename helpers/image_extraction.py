import pymupdf as fitz
import re
import os
import glob

class DiagramExtractor:
    def __init__(self,pdf_dir, output_dir, dpi=200):
        self.pdf_dir = pdf_dir
        self.output_dir = output_dir
        self.dpi = dpi
        os.makedirs(output_dir, exist_ok=True)
        self.pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        # Matches: "Figure 1", "Fig. 2", "Figure 3:", etc.
        self.caption_pattern = re.compile(r'\b(Figure|Fig\.)\s+\d+', re.IGNORECASE)
        self.diagram_count = 0
    def extract_diagrams_by_captions(self):
        os.makedirs(self.output_dir, exist_ok=True)
        pdf_files = glob.glob(os.path.join(self.pdf_dir, "*.pdf"))
        
        # Matches: "Figure 1", "Fig. 2", "Figure 3:", etc.
        caption_pattern = re.compile(r'\b(Figure|Fig\.)\s+\d+', re.IGNORECASE)
        self.diagram_count = 0

        for pdf_path in pdf_files:
            pdf_name = os.path.basename(pdf_path).replace(".pdf", "")
            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                page = doc[page_num]
                text_instances = page.get_text("blocks")  # (x0, y0, x1, y1, "text", block_no, block_type)

                for block in text_instances:
                    text = block[4]
                    if caption_pattern.search(text):
                        # Found a caption block!
                        caption_rect = fitz.Rect(block[:4])
                        
                        # Assume the diagram is in the space above the caption (typical for academic papers)
                        # You can adjust the height (e.g., 300-400 points above caption)
                        crop_box = fitz.Rect(
                            max(0, page.rect.x0),                      # Left margin
                            max(0, caption_rect.y0 - 350),             # Top of diagram
                            min(page.rect.x1, page.rect.width),        # Right margin
                            min(page.rect.y1, caption_rect.y1 + 40)    # Include caption text
                        )

                        # Render only that cropped region at high resolution
                        pix = page.get_pixmap(dpi=self.dpi, clip=crop_box)
                        
                        output_file = os.path.join(
                            self.output_dir, 
                            f"{pdf_name}_p{page_num+1}_fig_{self.diagram_count+1}.png"
                        )
                        pix.save(output_file)
                        self.diagram_count += 1

            doc.close()

        print(f"Extracted {self.diagram_count} diagrams based on figure captions.")