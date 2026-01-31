"""
PDF Generation Utility for Research Papers
==========================================

Author: Bhavin Kotak
Work Email: bhavin.kotak@ibm.com
Personal Email: kotak.bhavin@gmail.com

Generates PDF documents from Markdown files using multiple methods.
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

try:
    import markdown
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

try:
    from markdown_pdf import MarkdownPdf, Section
    MARKDOWN_PDF_AVAILABLE = True
except ImportError:
    MARKDOWN_PDF_AVAILABLE = False


class PDFGenerator:
    """Generate PDFs from Markdown files"""
    
    def __init__(self, output_dir: str = "pdfs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_pdf_weasyprint(self, markdown_path: str, output_path: str = None) -> str:
        """
        Generate PDF using WeasyPrint (high quality, supports CSS).
        
        Args:
            markdown_path: Path to markdown file
            output_path: Optional custom output path
            
        Returns:
            Path to generated PDF
        """
        if not WEASYPRINT_AVAILABLE:
            raise ImportError("WeasyPrint not available. Install with: pip install weasyprint markdown")
        
        # Read markdown
        with open(markdown_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert to HTML
        html_content = markdown.markdown(
            md_content,
            extensions=[
                'tables',
                'fenced_code',
                'codehilite',
                'toc',
                'meta'
            ]
        )
        
        # Wrap in full HTML document with styling
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Research Paper</title>
            <style>
                @page {{
                    size: A4;
                    margin: 2cm;
                }}
                body {{
                    font-family: 'Times New Roman', Times, serif;
                    font-size: 11pt;
                    line-height: 1.6;
                    color: #333;
                    max-width: 100%;
                }}
                h1 {{
                    font-size: 18pt;
                    text-align: center;
                    margin-bottom: 0.5cm;
                    color: #000;
                }}
                h2 {{
                    font-size: 14pt;
                    border-bottom: 1px solid #ccc;
                    padding-bottom: 0.2cm;
                    margin-top: 1cm;
                }}
                h3 {{
                    font-size: 12pt;
                    margin-top: 0.8cm;
                }}
                p {{
                    text-align: justify;
                    margin-bottom: 0.5cm;
                }}
                code {{
                    font-family: 'Courier New', monospace;
                    font-size: 9pt;
                    background-color: #f5f5f5;
                    padding: 0.1cm 0.2cm;
                    border-radius: 3px;
                }}
                pre {{
                    background-color: #f5f5f5;
                    padding: 0.5cm;
                    border-radius: 5px;
                    overflow-x: auto;
                    font-size: 9pt;
                    line-height: 1.4;
                }}
                pre code {{
                    background: none;
                    padding: 0;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 1cm 0;
                    font-size: 10pt;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 0.3cm;
                    text-align: left;
                }}
                th {{
                    background-color: #f0f0f0;
                    font-weight: bold;
                }}
                blockquote {{
                    border-left: 3px solid #ccc;
                    padding-left: 0.5cm;
                    margin-left: 0;
                    font-style: italic;
                    color: #666;
                }}
                strong {{
                    color: #000;
                }}
                ul, ol {{
                    margin-bottom: 0.5cm;
                }}
                li {{
                    margin-bottom: 0.2cm;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Generate output path
        if output_path is None:
            input_name = Path(markdown_path).stem
            output_path = self.output_dir / f"{input_name}.pdf"
        
        # Generate PDF
        HTML(string=full_html).write_pdf(str(output_path))
        
        return str(output_path)
    
    def generate_pdf_markdown_pdf(self, markdown_path: str, output_path: str = None) -> str:
        """
        Generate PDF using markdown-pdf package.
        
        Args:
            markdown_path: Path to markdown file
            output_path: Optional custom output path
            
        Returns:
            Path to generated PDF
        """
        if not MARKDOWN_PDF_AVAILABLE:
            raise ImportError("markdown-pdf not available. Install with: pip install markdown-pdf")
        
        # Generate output path
        if output_path is None:
            input_name = Path(markdown_path).stem
            output_path = str(self.output_dir / f"{input_name}.pdf")
        
        # Read markdown
        with open(markdown_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Create PDF
        pdf = MarkdownPdf()
        pdf.add_section(Section(md_content))
        pdf.save(output_path)
        
        return output_path
    
    def generate_pdf_pandoc(self, markdown_path: str, output_path: str = None) -> str:
        """
        Generate PDF using Pandoc (requires system installation).
        
        Args:
            markdown_path: Path to markdown file
            output_path: Optional custom output path
            
        Returns:
            Path to generated PDF
        """
        # Generate output path
        if output_path is None:
            input_name = Path(markdown_path).stem
            output_path = str(self.output_dir / f"{input_name}.pdf")
        
        # Run pandoc
        cmd = [
            'pandoc',
            markdown_path,
            '-o', output_path,
            '--pdf-engine=xelatex',
            '-V', 'geometry:margin=1in',
            '-V', 'fontsize=11pt',
            '--highlight-style=tango'
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Pandoc failed: {e.stderr.decode()}")
        except FileNotFoundError:
            raise RuntimeError("Pandoc not installed. Install from: https://pandoc.org/installing.html")
    
    def generate_pdf(self, markdown_path: str, method: str = 'auto') -> str:
        """
        Generate PDF using best available method.
        
        Args:
            markdown_path: Path to markdown file
            method: 'weasyprint', 'markdown-pdf', 'pandoc', or 'auto'
            
        Returns:
            Path to generated PDF
        """
        if method == 'auto':
            if WEASYPRINT_AVAILABLE:
                return self.generate_pdf_weasyprint(markdown_path)
            elif MARKDOWN_PDF_AVAILABLE:
                return self.generate_pdf_markdown_pdf(markdown_path)
            else:
                return self.generate_pdf_pandoc(markdown_path)
        elif method == 'weasyprint':
            return self.generate_pdf_weasyprint(markdown_path)
        elif method == 'markdown-pdf':
            return self.generate_pdf_markdown_pdf(markdown_path)
        elif method == 'pandoc':
            return self.generate_pdf_pandoc(markdown_path)
        else:
            raise ValueError(f"Unknown method: {method}")


def generate_all_pdfs(papers_dir: str, output_dir: str = "pdfs"):
    """
    Generate PDFs for all markdown papers in a directory.
    
    Args:
        papers_dir: Directory containing markdown files
        output_dir: Output directory for PDFs
    """
    generator = PDFGenerator(output_dir)
    papers_path = Path(papers_dir)
    
    if not papers_path.exists():
        print(f"Error: Directory {papers_dir} not found")
        return
    
    markdown_files = list(papers_path.glob("*.md"))
    
    if not markdown_files:
        print(f"No markdown files found in {papers_dir}")
        return
    
    print(f"\nGenerating PDFs for {len(markdown_files)} files...")
    print("=" * 60)
    
    successful = []
    failed = []
    
    for md_file in markdown_files:
        print(f"\nProcessing: {md_file.name}")
        try:
            pdf_path = generator.generate_pdf(str(md_file))
            print(f"  ✓ Generated: {pdf_path}")
            successful.append(md_file.name)
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            failed.append((md_file.name, str(e)))
    
    print("\n" + "=" * 60)
    print(f"Summary: {len(successful)} successful, {len(failed)} failed")
    
    if failed:
        print("\nFailed files:")
        for name, error in failed:
            print(f"  - {name}: {error}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate PDFs from Markdown files")
    parser.add_argument("input", help="Input markdown file or directory")
    parser.add_argument("-o", "--output", help="Output directory", default="pdfs")
    parser.add_argument("-m", "--method", choices=['auto', 'weasyprint', 'markdown-pdf', 'pandoc'],
                       default='auto', help="PDF generation method")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        generator = PDFGenerator(args.output)
        pdf_path = generator.generate_pdf(str(input_path), method=args.method)
        print(f"Generated: {pdf_path}")
    elif input_path.is_dir():
        generate_all_pdfs(str(input_path), args.output)
    else:
        print(f"Error: {args.input} not found")
        sys.exit(1)
