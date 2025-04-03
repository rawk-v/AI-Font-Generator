import cv2 # OpenCV for image processing
import numpy as np # NumPy for numerical operations
from PIL import Image # Pillow for image handling (opening/saving non-OpenCV formats)
import os
import string
import math
import traceback # For detailed error printing

# Removed unused sort_contours function

def split_font_grid_contours(image_path, output_dir, characters, padding=5):
    """
    Splits a font glyph grid image into individual character images using contour detection.
    Attempts to use transparency (alpha channel) if available, otherwise falls back
    to grayscale adaptive thresholding.

    Args:
        image_path (str): Path to the input grid image file (PNG preferred for transparency).
        output_dir (str): Directory where the character images will be saved.
        characters (str): A string containing all characters expected in the grid,
                          in reading order (left-to-right, top-to-bottom).
        padding (int): Pixels to add around the detected bounding box when cropping.
    """
    try:
        # --- 1. Load Image (Attempting to preserve Alpha channel) ---
        img_pil = None
        img_cv = None
        has_alpha = False
        try:
            # Try loading with Pillow first, as it often handles formats like PNG well
            img_pil = Image.open(image_path)
            print(f"Pillow loaded image mode: {img_pil.mode}")

            if 'A' in img_pil.mode: # Check if Pillow detected an Alpha channel
                has_alpha = True
                # Convert Pillow RGBA/LA to OpenCV BGRA format
                # Ensure correct conversion based on mode (RGBA or LA)
                if img_pil.mode == 'RGBA':
                    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGRA)
                elif img_pil.mode == 'LA': # Grayscale + Alpha
                    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_LA2BGRA)
                else: # Fallback for other alpha modes if necessary
                     img_cv = cv2.cvtColor(np.array(img_pil.convert('RGBA')), cv2.COLOR_RGBA2BGRA)
                print("Loaded image with Alpha channel using Pillow -> OpenCV.")
            else:
                 # Convert Pillow RGB/L to OpenCV BGR format
                img_cv = cv2.cvtColor(np.array(img_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
                print("Loaded image without Alpha channel using Pillow -> OpenCV.")

        except FileNotFoundError:
            print(f"Error: Input image file not found at {image_path}")
            return
        except Exception as e:
            print(f"Error loading image with Pillow: {e}. Trying direct OpenCV load.")
            # Fallback to cv2.imread if Pillow fails
            img_cv = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # Load with alpha if possible
            if img_cv is None:
                print(f"Error: Could not load image at {image_path} using OpenCV either.")
                return

            # Check channels after loading with OpenCV
            if len(img_cv.shape) == 3 and img_cv.shape[2] == 4:
                has_alpha = True
                print("Loaded image with Alpha channel using direct OpenCV.")
            elif len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
                 has_alpha = False
                 print("Loaded image as 3-channel BGR using direct OpenCV.")
            elif len(img_cv.shape) == 2: # Grayscale loaded
                 has_alpha = False
                 img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR) # Convert to BGR
                 print("Loaded image as Grayscale using direct OpenCV, converted to BGR.")
            else:
                 print(f"Warning: Loaded image has unexpected shape {img_cv.shape}. Attempting to proceed.")
                 # Attempt conversion to BGR if possible
                 try:
                     img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
                     has_alpha = False
                 except:
                      print("Error: Could not convert image to standard BGR format.")
                      return


        img_height, img_width, channels = img_cv.shape
        print(f"Image size: {img_width}x{img_height}, Channels: {channels}, Has Alpha: {has_alpha}")

        # --- 2. Preprocessing (Create Binary Mask) ---
        thresh = None # Initialize thresh
        if has_alpha and channels == 4: # Ensure we actually have 4 channels before accessing index 3
            print("Using Alpha channel for thresholding.")
            # Extract alpha channel (channel 3 in BGRA)
            alpha_channel = img_cv[:, :, 3]
            # Threshold the alpha channel: pixels with alpha > 10 are considered foreground
            # Adjust the threshold (10) if needed based on your image's transparency levels
            _, thresh = cv2.threshold(alpha_channel, 10, 255, cv2.THRESH_BINARY)
        else:
            print("No Alpha channel or unexpected channel count. Using grayscale adaptive thresholding.")
            # Ensure image is 3 channels BGR before converting to gray
            if channels == 4: # Handle case where alpha detected but not used (e.g., Pillow load failed)
                 img_cv_bgr = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)
            elif channels == 3:
                 img_cv_bgr = img_cv
            else: # Should have been handled in loading, but as a safeguard
                 print("Error: Cannot convert to grayscale due to unexpected channel count.")
                 return

            # Convert to grayscale
            gray = cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGR2GRAY)
            # Apply adaptive thresholding
            # *** ADJUST THESE PARAMETERS AS NEEDED ***
            blockSize = 11 # Must be odd
            C = 5 # Constant to subtract
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, blockSize, C)

        if thresh is None:
             print("Error: Threshold image could not be generated.")
             return

        # Optional: Save the threshold image for debugging
        # Ensure output directory exists before saving debug image
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")
            except OSError as e:
                print(f"Error creating directory {output_dir}: {e}")
                return # Cannot proceed without output directory
        cv2.imwrite(os.path.join(output_dir, "_debug_threshold.png"), thresh)

        # Optional: Apply morphological operations to remove noise/fill gaps
        # *** ADJUST KERNEL SIZE OR UNCOMMENT AS NEEDED ***
        # kernel_size = 2
        # kernel = np.ones((kernel_size, kernel_size),np.uint8)
        # # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel) # Remove noise
        # # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel) # Fill gaps

        # --- 3. Contour Finding ---
        # *** KEY CHANGE HERE: Use cv2.RETR_LIST instead of cv2.RETR_EXTERNAL ***
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Found {len(contours)} initial contours (using RETR_LIST).")

        # --- 4. Filtering Contours ---
        valid_contours_boxes = []
        # *** ADJUST THESE FILTERING PARAMETERS BASED ON YOUR IMAGE/FONT SIZE ***
        min_char_area = 50       # Min area in pixels
        # Heuristic max area calculation - adjust grid numbers (6x5) if your grid is different
        avg_cell_width = img_width / 6
        avg_cell_height = img_height / 5
        max_char_area = avg_cell_width * avg_cell_height * 0.9 # Heuristic: 90% of avg cell area
        min_char_height = 10     # Min height in pixels
        min_char_width = 5       # Min width in pixels

        print(f"Filtering contours with: min_area={min_char_area:.0f}, max_area={max_char_area:.0f}, min_height={min_char_height}, min_width={min_char_width}")

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)

            # Filter based on area and dimensions
            if min_char_area < area < max_char_area and h > min_char_height and w > min_char_width:
                 valid_contours_boxes.append({'contour': cnt, 'box': (x, y, w, h), 'area': area})
            # else:
            #      # Uncomment to debug skipped contours
            #      print(f"  - Skipping contour: Area={area:.1f}, Pos=({x},{y}), Size=({w}x{h})")

        print(f"Found {len(valid_contours_boxes)} potentially valid character contours after filtering.")

        if not valid_contours_boxes:
            print("Error: No valid character contours found after filtering. Check image, thresholding, and filtering parameters.")
            # Provide more specific advice based on the number of initial contours found
            if len(contours) > 1:
                 print("Suggestion: Initial contours were found, so the filtering parameters (min/max area/size) might be too strict or incorrect for your font.")
            elif len(contours) <= 1:
                 print("Suggestion: Very few initial contours found. Check the _debug_threshold.png image. Ensure characters are clearly separated from the background. If using Alpha, check the alpha channel itself. If using grayscale, adjust adaptiveThreshold parameters.")
            return

        # --- 5. Sorting Contours/Bounding Boxes ---
        # Sort primarily by Y (top), then by X (left) to handle reading order
        valid_contours_boxes.sort(key=lambda item: (item['box'][1], item['box'][0])) # Sort by y, then x

        num_expected_chars = len(characters)
        num_found_chars = len(valid_contours_boxes)
        if num_found_chars != num_expected_chars:
             print(f"Warning: Found {num_found_chars} contours, but expected {num_expected_chars} characters based on CHARACTERS string. Output may be incomplete or contain extras. Check filtering or CHARACTERS string.")


        # --- 6. Cropping and Saving ---
        # Output directory should exist from saving debug image, but double check
        if not os.path.exists(output_dir):
             os.makedirs(output_dir) # Create it if it somehow doesn't exist

        char_index = 0
        for i, item in enumerate(valid_contours_boxes):
            if char_index < len(characters):
                x, y, w, h = item['box']

                # Get the character for the current contour
                char = characters[char_index]

                # Define crop coordinates with padding (ensure bounds are valid)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(img_width, x + w + padding)
                y2 = min(img_height, y + h + padding)

                # Crop from the *original* image (BGRA or BGR)
                # Ensure slicing indices are integers
                y1, y2, x1, x2 = int(y1), int(y2), int(x1), int(x2)
                char_img_cv = img_cv[y1:y2, x1:x2]

                # --- Create a safe filename ---
                filename_map = {',': 'comma', '.': 'period', '?': 'question_mark', '!': 'exclamation_mark'}
                safe_char_name = filename_map.get(char, char)
                valid_filename_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
                filename_base = ''.join(c if c in valid_filename_chars else '_' for c in safe_char_name)
                if not filename_base: # Handle cases where the name becomes empty
                    filename_base = f"char_{char_index:02d}" # Use index if name is invalid

                output_filename = os.path.join(output_dir, f"{filename_base}.png")

                # Save using OpenCV imwrite - it handles PNG transparency automatically if char_img_cv has 4 channels
                try:
                    # Check if cropped image is empty before saving
                    if char_img_cv.size == 0:
                        print(f"Warning: Cropped image for '{char}' is empty. Skipping save for {output_filename}")
                        # Don't increment char_index if we skip saving, so subsequent chars match contours
                        continue # Skip to next contour

                    success = cv2.imwrite(output_filename, char_img_cv)
                    if success:
                        print(f"Saved: {output_filename} (Contour {i+1}/{len(valid_contours_boxes)})")
                    else:
                        # Provide more details if saving fails
                        print(f"Error: Failed to save {output_filename} using cv2.imwrite. Check permissions and path.")
                except Exception as save_e:
                     print(f"Error saving {output_filename} with cv2.imwrite: {save_e}")


                char_index += 1 # Increment character index only after successful processing/saving attempt
            else:
                # This case handles having more contours than expected characters
                print(f"Warning: More contours found ({i+1}) than characters specified ({len(characters)}). Stopping processing additional contours.")
                break

        # Final check on counts
        if char_index < num_expected_chars:
            print(f"\nWarning: Processed/saved {char_index} contours, but expected {num_expected_chars} characters. Some might have been missed due to filtering or errors.")
        elif num_found_chars > num_expected_chars:
             print(f"\nNote: Found {num_found_chars} contours initially, more than the {num_expected_chars} expected characters. Only the first {num_expected_chars} were processed.")


        print("\nProcessing complete.")

    except ImportError:
        print("Error: OpenCV, NumPy or Pillow not installed. Please install using:")
        print("pip install opencv-python numpy Pillow")
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
        traceback.print_exc() # Print detailed traceback for debugging


# --- Configuration ---
# *** USE THE ACTUAL PATH TO YOUR IMAGE FILE ***
INPUT_IMAGE = '/Users/rawk/Downloads/doodle-font2.png'
OUTPUT_FOLDER = 'output_characters_contours_alpha' # Folder to save the individual characters
# Characters in the grid (left-to-right, top-to-bottom)
# *** MAKE SURE THIS MATCHES YOUR IMAGE EXACTLY ***
CHARACTERS = "abcdefghijklmnopqrstuvwxyz,.?!"
# Padding around the detected character bounding box
# *** ADJUST PADDING AS NEEDED ***
PADDING = 5 # Pixels

# --- Run the function ---
split_font_grid_contours(INPUT_IMAGE, OUTPUT_FOLDER, CHARACTERS, PADDING)

