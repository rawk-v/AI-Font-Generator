# --- [ Imports and prepare_bitmap_data remain the same ] ---
import os
from pathlib import Path

import numpy as np
from PIL import Image
import potrace  # pypotrace, Python bindings for potrace
import math
import argparse

from typing import List

DEFAULT_INPUT_DIR = './output/characters/' # Directory containing a.png, b.png, comma.png etc.
DEFAULT_TURD_SIZE = 2       # Potrace option: suppress speckles smaller than this pixel size
DEFAULT_OPT_TOLERANCE = 0.2 # Potrace option: curve optimization tolerance
DEFAULT_ALPHAMAX = 1.0
DEFAULT_OPTICURVE = 1

def prepare_bitmap_data(png_path, threshold=128):
    """
    Loads a PNG file (expecting black glyph on transparent background),
    converts it to a NumPy boolean array (True for glyph pixels)
    suitable for pypotrace.
    """
    try:
        img = Image.open(png_path).convert('RGBA')
    except FileNotFoundError:
        print(f"Error: Input PNG not found at {png_path}")
        return None
    except Exception as e:
        print(f"Error opening image {png_path}: {e}")
        return None

    # Create a white background image
    bg = Image.new('RGB', img.size, (255, 255, 255))
    # Paste the glyph using its alpha channel as a mask
    bg.paste(img, mask=img.split()[3])

    # Convert to grayscale, then to black/white bitmap (mode '1')
    # Pixels darker than threshold become black (0), others white (255)
    bw_img = bg.convert('L').point(lambda x: 0 if x < threshold else 255, '1')

    # Convert Pillow '1' mode image to NumPy array
    # Invert it so that black pixels (glyph) are True/1 and white pixels are False/0
    bitmap_data = np.invert(np.array(bw_img))
    return bitmap_data


def path_to_svg_d(path):
    """
    Converts a pypotrace Path object into an SVG path 'd' string.
    """
    d_parts = []
    if not path:
        return ""

    for curve in path:
        start_pt = curve.start_point
        d_parts.append(f"M{start_pt[0]:.3f},{start_pt[1]:.3f}")

        for segment in curve.segments:
            end_pt = segment.end_point
            if segment.is_corner:
                d_parts.append(f"L{end_pt[0]:.3f},{end_pt[1]:.3f}")
            else:
                c1 = segment.c1
                c2 = segment.c2
                d_parts.append(f"C{c1[0]:.3f},{c1[1]:.3f} "
                               f"{c2[0]:.3f},{c2[1]:.3f} "
                               f"{end_pt[0]:.3f},{end_pt[1]:.3f}")
        d_parts.append("Z") # Close the path

    return " ".join(d_parts)

def calculate_viewbox(path, img_width, img_height, margin=1):
    """
    Calculates the viewBox for the SVG based on the path bounds.
    """
    fallback_viewbox = (0, 0, img_width, img_height)
    if not path:
        return fallback_viewbox

    # Use min_y, max_y directly assuming Y-down coordinates
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    def update_bounds(pt):
        nonlocal min_x, min_y, max_x, max_y
        min_x = min(min_x, pt[0])
        max_x = max(max_x, pt[0])
        min_y = min(min_y, pt[1])
        max_y = max(max_y, pt[1])

    for curve in path:
        # curve.start_point is a tuple
        update_bounds(curve.start_point)
        for segment in curve.segments:
            # segment.end_point is a tuple
            update_bounds(segment.end_point)
            if not segment.is_corner:
                # segment.c1 and segment.c2 are tuples
                update_bounds(segment.c1)
                update_bounds(segment.c2)

    # Handle cases where path might be empty or degenerate
    if min_x == float('inf'):
        return fallback_viewbox

    # Calculate viewBox attributes with margin
    vb_x = math.floor(min_x - margin)
    vb_y = math.floor(min_y - margin)
    vb_w = math.ceil(max_x - min_x + 2 * margin)
    vb_h = math.ceil(max_y - min_y + 2 * margin)

    # Ensure width and height are positive
    vb_w = max(1, vb_w)
    vb_h = max(1, vb_h)

    return vb_x, vb_y, vb_w, vb_h

def convert_png_to_svg(images: List[str], **kwargs):
    """
    Converts a list of PNG files to SVG format.
    
    Args:
        images: List of paths to PNG files
        **kwargs: Optional parameters for potrace
        
    """
    turdsize = kwargs.get('turdsize', DEFAULT_TURD_SIZE)
    opttolerance = kwargs.get('opttolerance', DEFAULT_OPT_TOLERANCE)
    alphamax = kwargs.get('alphamax', DEFAULT_ALPHAMAX)
    opticurve = kwargs.get('opticurve', DEFAULT_OPTICURVE)
    
    processed_files = 0
    
    for png_path in images:
        print(f"Processing {png_path}...")

        # 1. Prepare Bitmap Data
        bitmap_data = prepare_bitmap_data(png_path)
        if bitmap_data is None:
            print(f"  Skipping {png_path} due to loading error.")
            continue

        img_height, img_width = bitmap_data.shape[0:2]

        # 2. Trace using pypotrace
        try:
            bitmap = potrace.Bitmap(bitmap_data)
            path = bitmap.trace(
                turdsize=turdsize,
                opttolerance=opttolerance,
                alphamax=alphamax,
                opticurve=opticurve
            )
        except Exception as e:
            print(f"  Error during tracing for {png_path}: {e}")
            continue

        # 3. Generate SVG Content
        if not path:
            print(f"  Warning: No path traced for {png_path}. Skipping SVG generation.")
            continue

        vb_x, vb_y, vb_w, vb_h = calculate_viewbox(path, img_width, img_height)
        path_d = path_to_svg_d(path)

        viewbox_str = f"{vb_x} {vb_y} {vb_w} {vb_h}"

        svg_content = f'''<?xml version="1.0" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg width="{vb_w}px" height="{vb_h}px" viewBox="{viewbox_str}"
     xmlns="http://www.w3.org/2000/svg" version="1.1">
  <path d="{path_d}" fill="black" stroke="none"/>
</svg>
'''
        # 4. Save SVG File
        svg_path = Path(png_path).with_suffix('.svg').as_posix()
        try:
            with open(svg_path, 'w') as f:
                f.write(svg_content)
            print(f"  -> Saved {svg_path}")
            processed_files += 1
        except Exception as e:
            print(f"  Error writing SVG file {svg_path}: {e}")
    print(f"Finished processing {len(images)} files. Converted {processed_files} to SVG.")


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Convert character PNG glyphs to SVG using pypotrace.")
    parser.add_argument(
        "-i", "--input_dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory containing character PNG files (default: {DEFAULT_INPUT_DIR})"
    )
    parser.add_argument(
        "-t", "--turdsize", type=int,
        default=DEFAULT_TURD_SIZE,
        help=f"Potrace 'turdsize' parameter (suppress speckles) (default: {DEFAULT_TURD_SIZE})"
    )
    parser.add_argument(
        "-O", "--opttolerance", type=float, 
        default=DEFAULT_OPT_TOLERANCE,
        help=f"Potrace 'opttolerance' parameter (curve optimization) (default: {DEFAULT_OPT_TOLERANCE})"
    )
    parser.add_argument(
        "-a", "--alphamax", type=float,
        default=DEFAULT_ALPHAMAX,
        help=f"Potrace 'alphamax' parameter (corner smoothing) (default: {DEFAULT_ALPHAMAX})"
    )
    parser.add_argument(
        "--opticurve", type=int, choices=[0, 1],
        default=DEFAULT_OPTICURVE,
        help=f"Potrace 'opticurve' parameter (enable/disable curve optimization) (default: {DEFAULT_OPTICURVE})"
    )
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        exit(1)
    
    print(f"Input directory: {input_dir}")
    print(f"Using Potrace parameters: turdsize={args.turdsize}, opttolerance={args.opttolerance}, alphamax={args.alphamax}, opticurve={args.opticurve}")
    
    png_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(".png")]
    
    # Process the PNG files
    convert_png_to_svg(
        png_files,
        turdsize=args.turdsize,
        opttolerance=args.opttolerance,
        alphamax=args.alphamax,
        opticurve=args.opticurve
    )
