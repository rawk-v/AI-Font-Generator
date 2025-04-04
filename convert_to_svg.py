import os
import numpy as np
from PIL import Image
import potrace
import math
import argparse

DEFAULT_INPUT_DIR = './output/characters/' # Directory containing a.png, b.png, comma.png etc.
DEFAULT_OUTPUT_DIR = './output_svgs/'
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


def path_to_svg_d(path, img_height):
    """
    Converts a pypotrace Path object into an SVG path 'd' string.
    """
    d_parts = []
    if not path:
        return []

    for curve in path:
        start_pt = curve.start_point
        d_parts.append(f"M{start_pt[0]:.3f},{img_height - start_pt[1]:.3f}")

        for segment in curve.segments:
            # segment.end_point is a tuple
            end_pt = segment.end_point
            if segment.is_corner:
                # A corner segment means a line to the end_point from the previous point.
                # FIX: Access tuple elements by index
                d_parts.append(f"L{end_pt[0]:.3f},{img_height - end_pt[1]:.3f}")
            else: # is_bezier
                # segment.c1, segment.c2 are tuples
                c1 = segment.c1
                c2 = segment.c2
                # FIX: Access tuple elements by index
                d_parts.append(f"C{c1[0]:.3f},{img_height - c1[1]:.3f} "
                               f"{c2[0]:.3f},{img_height - c2[1]:.3f} "
                               f"{end_pt[0]:.3f},{img_height - end_pt[1]:.3f}")
        d_parts.append("Z") # Close the path

    return d_parts

def calculate_viewbox(path, img_height, margin=1):
    """
    Calculates the viewBox for the SVG based on the path bounds.
    """
    if not path:
        return 0, 0, img_height, img_height

    min_x, min_y_svg = float('inf'), float('inf')
    max_x, max_y_svg = float('-inf'), float('-inf')

    # Function to update bounds, applying Y flip
    def update_bounds(pt):
        # FIX: Access tuple elements by index (pt[0] is x, pt[1] is y)
        nonlocal min_x, min_y_svg, max_x, max_y_svg
        svg_y = img_height - pt[1] # Use pt[1] for y
        min_x = min(min_x, pt[0])  # Use pt[0] for x
        max_x = max(max_x, pt[0])  # Use pt[0] for x
        min_y_svg = min(min_y_svg, svg_y)
        max_y_svg = max(max_y_svg, svg_y)

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
        return f"0 0 {img_height} {img_height}" # Fallback

    # Calculate viewBox attributes with margin
    vb_x = math.floor(min_x - margin)
    vb_y = math.floor(min_y_svg - margin)
    vb_w = math.ceil(max_x - min_x + 2 * margin)
    vb_h = math.ceil(max_y_svg - min_y_svg + 2 * margin)

    # Ensure width and height are positive
    vb_w = max(1, vb_w)
    vb_h = max(1, vb_h)

    return vb_x, vb_y, vb_w, vb_h


# --- Main Execution ---
def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Convert character PNG glyphs to SVG using pypotrace.")
    parser.add_argument(
        "-i", "--input_dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory containing character PNG files (default: {DEFAULT_INPUT_DIR})"
    )
    parser.add_argument(
        "-o", "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for SVG files (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "-t", "--turdsize", type=int,
        default=DEFAULT_TURD_SIZE,
        help=f"Potrace 'turdsize' parameter (suppress speckles) (default: {DEFAULT_TURD_SIZE})"
    )
    parser.add_argument(
        "-O", "--opttolerance", type=float, # Changed to capital O to match potrace convention
        default=DEFAULT_OPT_TOLERANCE,
        help=f"Potrace 'opttolerance' parameter (curve optimization) (default: {DEFAULT_OPT_TOLERANCE})"
    )
    # Add alphamax argument if you want to control it
    parser.add_argument(
        "-a", "--alphamax", type=float,
        default=DEFAULT_ALPHAMAX,
        help=f"Potrace 'alphamax' parameter (corner smoothing) (default: {DEFAULT_ALPHAMAX})"
    )
    # Add opticurve argument if you want to control it (though default is likely fine)
    parser.add_argument(
        "--opticurve", type=int, choices=[0, 1],
        default=DEFAULT_OPTICURVE,
        help=f"Potrace 'opticurve' parameter (enable/disable curve optimization) (default: {DEFAULT_OPTICURVE})"
    )


    args = parser.parse_args()

    input_dir = args.input_dir
    output_base_dir = args.output_dir
    output_dir = os.path.join(output_base_dir, 'characters')

    # --- Directory Setup ---
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Using Potrace parameters: turdsize={args.turdsize}, opttolerance={args.opttolerance}, alphamax={args.alphamax}, opticurve={args.opticurve}")


    # --- Processing Loop ---
    processed_files = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".png"):
            char_name = os.path.splitext(filename)[0]
            png_path = os.path.join(input_dir, filename)
            svg_path = os.path.join(output_dir, f"{char_name}.svg")

            print(f"Processing {filename}...")

            # 1. Prepare Bitmap Data
            bitmap_data = prepare_bitmap_data(png_path)
            if bitmap_data is None:
                print(f"  Skipping {filename} due to loading error.")
                continue

            img_height = bitmap_data.shape[0] # Needed for Y-flipping

            # 2. Trace using pypotrace
            try:
                bitmap = potrace.Bitmap(bitmap_data)
                path = bitmap.trace(
                    turdsize=args.turdsize,
                    opttolerance=args.opttolerance,
                    alphamax=args.alphamax,
                    opticurve=args.opticurve
                    # turnpolicy=potrace.TURNPOLICY_MINORITY
                )
            except Exception as e:
                print(f"  Error during tracing for {filename}: {e}")
                continue

            # 3. Generate SVG Content
            if not path:
                 print(f"  Warning: No path traced for {filename}. Skipping SVG generation.")
                 continue

            viewbox = calculate_viewbox(path, img_height)
            path_d = path_to_svg_d(path, img_height)

            _, _, svg_width, svg_height = viewbox

            svg_content = f'''<?xml version="1.0" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg width="{svg_width}px" height="{svg_height}px" viewBox="{' '.join([str(i) for i in viewbox])}"
     xmlns="http://www.w3.org/2000/svg" version="1.1">
  <path d="{' '.join(path_d)}" fill="black" stroke="none"/>
</svg>
'''
            # 4. Save SVG File
            try:
                with open(svg_path, 'w') as f:
                    f.write(svg_content)
                print(f"  -> Saved {svg_path}")
                processed_files += 1
            except Exception as e:
                print(f"  Error writing SVG file {svg_path}: {e}")

    print(f"\nProcessing finished. Converted {processed_files} files.")

if __name__ == "__main__":
    main()