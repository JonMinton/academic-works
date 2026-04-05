#!/usr/bin/env python3
"""
Extract figures from PDF pages using Gemini's multimodal understanding.

Renders PDF pages as images, sends them to Gemini, and gets back
structured figure information: bounding boxes, captions, and figure IDs.

Usage:
    python3 scripts/gemini_extract_figures.py --pdf downloads/MintonThesis.pdf --pages 31 42 55
    python3 scripts/gemini_extract_figures.py --pdf downloads/MintonThesis.pdf --pages 135-162 --output phd/figures-ch8.json
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

def get_api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        env_file = Path.home() / ".env.gemini"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key:
        print("Error: GEMINI_API_KEY not found", file=sys.stderr)
        sys.exit(1)
    return key


def render_page_to_png(pdf_path, page_num, output_dir, dpi=200):
    """Render a single PDF page to PNG using pdftoppm."""
    output_prefix = output_dir / f"page-{page_num:03d}"
    output_file = Path(f"{output_prefix}-{page_num:03d}.png")

    # pdftoppm uses 1-based page numbers
    pdftoppm = Path.home() / "miniconda" / "bin" / "pdftoppm"
    if not pdftoppm.exists():
        pdftoppm = "pdftoppm"

    cmd = [
        str(pdftoppm), "-png", "-r", str(dpi),
        "-f", str(page_num), "-l", str(page_num),
        "-singlefile",
        str(pdf_path), str(output_prefix)
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    # pdftoppm with -singlefile outputs as prefix.png
    actual_output = Path(f"{output_prefix}.png")
    if actual_output.exists():
        return actual_output
    # fallback: check for the numbered version
    if output_file.exists():
        return output_file
    raise FileNotFoundError(f"Expected output at {actual_output}")


def analyze_page_with_gemini(image_path, page_num, api_key):
    """Send a page image to Gemini and get figure analysis."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    with open(image_path, "rb") as f:
        image_data = f.read()

    prompt = f"""Analyze this academic thesis page (page {page_num}).

For each figure, chart, diagram, or image on this page, provide:
1. figure_id: The figure number as referenced in the text (e.g. "Figure 8.3")
2. caption: The full caption text
3. figure_type: One of "chart", "diagram", "photograph", "table", "flowchart", "graph"
4. description: A brief description of what the figure shows
5. bounding_box: Approximate location as percentages [top, left, bottom, right] where 0,0 is top-left

If there are NO figures on this page, return an empty list.

Respond ONLY with valid JSON in this format:
{{
  "page": {page_num},
  "figures": [
    {{
      "figure_id": "Figure X.Y",
      "caption": "...",
      "figure_type": "...",
      "description": "...",
      "bounding_box": [top_pct, left_pct, bottom_pct, right_pct]
    }}
  ]
}}"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=image_data, mime_type="image/png"),
                    types.Part.from_text(text=prompt),
                ],
            )
        ],
    )

    # Parse JSON from response
    text = response.text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"page": page_num, "figures": [], "raw_response": text}


def parse_page_range(pages_str):
    """Parse page specifications like '31', '135-162', '31 42 55'."""
    pages = []
    for part in pages_str:
        if "-" in part:
            start, end = part.split("-")
            pages.extend(range(int(start), int(end) + 1))
        else:
            pages.append(int(part))
    return sorted(set(pages))


def main():
    parser = argparse.ArgumentParser(description="Extract figures from PDF using Gemini")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--pages", nargs="+", required=True, help="Page numbers or ranges (e.g. 31 42 135-162)")
    parser.add_argument("--output", help="Output JSON file (default: stdout)")
    parser.add_argument("--dpi", type=int, default=200, help="Rendering DPI (default: 200)")
    parser.add_argument("--render-dir", default="/tmp/gemini-pages", help="Directory for rendered pages")
    args = parser.parse_args()

    api_key = get_api_key()
    pages = parse_page_range(args.pages)

    render_dir = Path(args.render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for page_num in pages:
        print(f"Processing page {page_num}...", file=sys.stderr)

        # Render page
        try:
            img_path = render_page_to_png(args.pdf, page_num, render_dir, args.dpi)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"  Error rendering page {page_num}: {e}", file=sys.stderr)
            continue

        # Analyze with Gemini
        try:
            result = analyze_page_with_gemini(img_path, page_num, api_key)
            results.append(result)
            n_figs = len(result.get("figures", []))
            print(f"  Found {n_figs} figure(s)", file=sys.stderr)
        except Exception as e:
            print(f"  Error analyzing page {page_num}: {e}", file=sys.stderr)
            results.append({"page": page_num, "error": str(e)})

    output = {"pdf": args.pdf, "pages_analyzed": len(results), "results": results}

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
