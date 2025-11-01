#!/bin/bash

# Compile arXiv Paper Script
# Usage: ./compile_paper.sh

echo "🧠 Compiling Thinking Engine arXiv Paper"
echo "========================================"

# Check if pdflatex is installed
if ! command -v pdflatex &> /dev/null; then
    echo "❌ Error: pdflatex not found. Please install LaTeX:"
    echo "   macOS: brew install mactex"
    echo "   Ubuntu: sudo apt-get install texlive-latex-base texlive-fonts-recommended texlive-extra-utils texlive-latex-recommended"
    echo "   Windows: Install MiKTeX or TeX Live"
    exit 1
fi

# Check if bibtex is available
if ! command -v bibtex &> /dev/null; then
    echo "⚠️  Warning: bibtex not found. References will not be processed."
fi

echo "📄 Compiling LaTeX document..."

# First compilation
pdflatex -interaction=nonstopmode arxiv_paper.tex

# Run bibtex if available
if command -v bibtex &> /dev/null; then
    echo "📚 Processing bibliography..."
    bibtex arxiv_paper
fi

# Second compilation for references
pdflatex -interaction=nonstopmode arxiv_paper.tex

# Third compilation for final output
pdflatex -interaction=nonstopmode arxiv_paper.tex

echo "✅ Compilation complete!"

# Check if PDF was created
if [ -f "arxiv_paper.pdf" ]; then
    echo "📋 PDF generated: arxiv_paper.pdf"
    echo "📏 File size: $(ls -lh arxiv_paper.pdf | awk '{print $5}')"
    echo ""
    echo "🎉 Your arXiv paper is ready!"
    echo "   Open arxiv_paper.pdf to view your research paper"
else
    echo "❌ Error: PDF was not generated. Check for LaTeX errors above."
    exit 1
fi

echo ""
echo "🧹 Cleaning up auxiliary files..."
# Clean up auxiliary files (optional)
# rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot

echo "✨ Done! Your Thinking Engine research paper is complete."
