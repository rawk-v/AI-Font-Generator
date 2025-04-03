import cv2
import numpy as np
from PIL import Image
import os
import sys
import traceback # For detailed error printing

def split_font_grid_contours_indexed(image_path, output_dir, characters, padding=5, grid_rows=5, grid_cols=6):
    """
    Splits a font glyph grid image into individual character images using contour detection.
    Based on the user-preferred version, with added grid removal, indexed filenames,
    and additional intermediate debug image saving.
    Attempts to use transparency (alpha channel) if available, otherwise falls back
    to grayscale adaptive thresholding.

    Args:
        image_path (str): Path to the input grid image file (PNG preferred for transparency).
        output_dir (str): Directory where the character images will be saved.
        characters (str): A string containing all characters expected in the grid,
                          used for counting/warnings but NOT for filenames.
        padding (int): Pixels to add around the detected bounding box when cropping.
    """
    try:
        # --- 1. Load Image (Attempting to preserve Alpha channel) ---
        img_pil = None
        img_cv = None
        has_alpha = False
        # (Image loading logic - unchanged)
        try:
            img_pil = Image.open(image_path)
            print(f"Pillow loaded image mode: {img_pil.mode}")
            if 'A' in img_pil.mode:
                has_alpha = True
                if img_pil.mode == 'RGBA': img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGRA)
                elif img_pil.mode == 'LA': img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_LA2BGRA)
                else: img_cv = cv2.cvtColor(np.array(img_pil.convert('RGBA')), cv2.COLOR_RGBA2BGRA)
                print("Loaded image with Alpha channel using Pillow -> OpenCV.")
            else:
                img_cv = cv2.cvtColor(np.array(img_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
                print("Loaded image without Alpha channel using Pillow -> OpenCV.")
        except FileNotFoundError: print(f"Error: Input image file not found at {image_path}"); return
        except Exception as e:
            print(f"Error loading image with Pillow: {e}. Trying direct OpenCV load.")
            img_cv = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img_cv is None: print(f"Error: Could not load image at {image_path} using OpenCV either."); return
            if len(img_cv.shape) == 3 and img_cv.shape[2] == 4: has_alpha = True; print("Loaded image with Alpha channel using direct OpenCV.")
            elif len(img_cv.shape) == 3 and img_cv.shape[2] == 3: has_alpha = False; print("Loaded image as 3-channel BGR using direct OpenCV.")
            elif len(img_cv.shape) == 2: has_alpha = False; img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR); print("Loaded image as Grayscale using direct OpenCV, converted to BGR.")
            else:
                 print(f"Warning: Loaded image has unexpected shape {img_cv.shape}.")
                 try: img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR); has_alpha = False
                 except: print("Error: Could not convert image to standard BGR format."); return

        img_height, img_width, channels = img_cv.shape
        print(f"Image size: {img_width}x{img_height}, Channels: {channels}, Has Alpha: {has_alpha}")

        # Ensure output directory exists early
        if not os.path.exists(output_dir):
            try: os.makedirs(output_dir); print(f"Created output directory: {output_dir}")
            except OSError as e: print(f"Error creating directory {output_dir}: {e}"); return

        # --- 2. Preprocessing (Create Binary Mask) ---
        thresh = None
        # (Thresholding logic - unchanged)
        if has_alpha and channels == 4:
            print("Using Alpha channel for thresholding.")
            cv2.imshow("Alpha Channel", img_cv[:, :, 3])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            alpha_channel = img_cv[:, :, 3]
            _, thresh = cv2.threshold(alpha_channel, 10, 255, cv2.THRESH_BINARY)
        else:
            print("No Alpha channel or unexpected channel count. Using grayscale adaptive thresholding.")
            if channels == 4: img_cv_bgr = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)
            elif channels == 3: img_cv_bgr = img_cv
            else: print("Error: Cannot convert to grayscale due to unexpected channel count."); return
            gray = cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGR2GRAY)
            blockSize = 11; C = 5
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize, C)

        if thresh is None: print("Error: Threshold image could not be generated."); return
        cv2.imwrite(os.path.join(output_dir, "_debug_threshold.png"), thresh) # Save threshold image

        # --- 3. Contour Finding ---
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Found {len(contours)} initial contours (using RETR_LIST).")

        # --- DEBUG VIS 1: Draw ALL contours ---
        img_with_all_contours = img_cv.copy()
        cv2.drawContours(img_with_all_contours, contours, -1, (0, 255, 0), 1) # Draw all contours in green
        cv2.imwrite(os.path.join(output_dir, "_debug_1_all_contours.png"), img_with_all_contours)
        print("Saved: _debug_1_all_contours.png")

        # --- 4. Filtering Contours ---
        valid_contours_boxes = []
        # (Filtering logic - unchanged, adjust parameters as needed)
        min_char_area = 50
        avg_cell_width = img_width / grid_cols
        avg_cell_height = img_height / grid_rows if grid_cols > 0 else img_height / 5
        max_char_area = avg_cell_width * avg_cell_height * 0.9
        min_char_height = 10
        min_char_width = 5
        print(f"Filtering contours with: min_area={min_char_area:.0f}, max_area={max_char_area:.0f}, min_height={min_char_height}, min_width={min_char_width}")

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if min_char_area < area < max_char_area and h > min_char_height and w > min_char_width:
                 valid_contours_boxes.append({'contour': cnt, 'box': (x, y, w, h), 'area': area})

        print(f"Found {len(valid_contours_boxes)} potentially valid character contours after filtering.")

        # --- DEBUG VIS 2: Draw FILTERED contours and boxes ---
        img_with_filtered_contours = img_cv.copy()
        for item in valid_contours_boxes:
             x, y, w, h = item['box']
             cv2.drawContours(img_with_filtered_contours, [item['contour']], -1, (255, 0, 0), 1) # Filtered contours in blue
             cv2.rectangle(img_with_filtered_contours, (x, y), (x + w, y + h), (0, 0, 255), 1) # Bounding boxes in red
        cv2.imwrite(os.path.join(output_dir, "_debug_2_filtered_contours_boxes.png"), img_with_filtered_contours)
        print("Saved: _debug_2_filtered_contours_boxes.png")


        if not valid_contours_boxes: print("Error: No valid character contours found after filtering."); return

        # --- 5. Sorting Contours/Bounding Boxes ---
        # (Simple (y,x) sorting - unchanged)
        print(f"Sorted {len(valid_contours_boxes)} contours using simple y,x sort.")

        valid_contours_boxes.sort(key=lambda item: item['area'], reverse=True)
        print(f"Sorted {len(valid_contours_boxes)} contours by area (largest to smallest).")

        # Remove duplicates (boxes that are contained within larger boxes)
        def is_duplicate(box1, box2, overlap_threshold=0.7):
            """Check if box2 is largely contained within box1"""
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            
            # Calculate intersection
            x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            intersection = x_overlap * y_overlap
            area2 = w2 * h2
            
            # If box2 is mostly contained within box1, consider it a duplicate
            return intersection > 0 and intersection / area2 >= overlap_threshold

        filtered_contours_boxes = []
        for i, item in enumerate(valid_contours_boxes):
            is_dup = False
            for j in range(i):
                if is_duplicate(valid_contours_boxes[j]['box'], item['box']):
                    is_dup = True
                    break
            if not is_dup:
                filtered_contours_boxes.append(item)

        print(f"Removed {len(valid_contours_boxes) - len(filtered_contours_boxes)} duplicate contours.")
        valid_contours_boxes = filtered_contours_boxes
        valid_contours_boxes.sort(key=lambda item: (item['box'][1], item['box'][0])) # Sort by y, then x
    
        # --- DEBUG VIS 3: Draw SORTED contours/boxes with index numbers ---
        img_with_sorted_contours = img_cv.copy()
        for i, item in enumerate(valid_contours_boxes):
             x, y, w, h = item['box']
             cv2.drawContours(img_with_sorted_contours, [item['contour']], -1, (255, 0, 0), 1) # Blue contours
             cv2.rectangle(img_with_sorted_contours, (x, y), (x + w, y + h), (0, 0, 255), 1) # Red boxes
             cv2.putText(img_with_sorted_contours, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Black index number above box
        cv2.imwrite(os.path.join(output_dir, "_debug_3_sorted_contours_boxes.png"), img_with_sorted_contours)
        print("Saved: _debug_3_sorted_contours_boxes.png")


        num_expected_chars = len(characters) if characters else 0
        num_found_chars = len(valid_contours_boxes)
        if characters and num_found_chars != num_expected_chars:
             print(f"Warning: Found/Sorted {num_found_chars} contours, but expected {num_expected_chars} characters based on CHARACTERS string.")

        # --- 6. Cropping, Grid Removal, and Saving ---
        # Calculate standard dimensions for all output images
        max_width = 0
        max_height = 0
        
        # First pass to determine max dimensions
        for item in valid_contours_boxes[:min(len(valid_contours_boxes), num_expected_chars if characters else len(valid_contours_boxes))]:
            _, _, w, h = item['box']
            # Add padding to dimensions
            w_padded = w + padding * 2
            h_padded = h + padding * 2
            max_width = max(max_width, w_padded)
            max_height = max(max_height, h_padded)
            
        # Round up to even numbers for better centering
        max_width = (max_width + 1) // 2 * 2
        max_height = (max_height + 1) // 2 * 2
        
        print(f"Standardizing all character output images to size: {max_width}x{max_height}")
        
        # Second pass to crop, standardize, and save
        for i, item in enumerate(valid_contours_boxes):
            if characters and i >= num_expected_chars:
                print(f"Warning: Found more contours ({num_found_chars}) than expected characters ({num_expected_chars}). Stopping at index {i}.")
                break

            x, y, w, h = item['box']
            contour = item['contour']

            # Define Crop Area (Bounding box only - tight crop)
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(img_width, x + w)
            y2 = min(img_height, y + h)
            y1, y2, x1, x2 = int(y1), int(y2), int(x1), int(x2)

            # Crop from the original image
            char_img_cv = img_cv[y1:y2, x1:x2]

            if char_img_cv.size == 0:
                print(f"Warning: Cropped image for index {i} is empty (coords: {y1}:{y2}, {x1}:{x2}). Skipping.")
                continue

            # Create mask for the character (more aggressive border removal)
            crop_h, crop_w = char_img_cv.shape[:2]
            mask = None

            if has_alpha and channels == 4 and char_img_cv.shape[2] == 4:
                # For transparent images, use alpha channel with more aggressive threshold
                cropped_alpha = char_img_cv[:, :, 3]
                _, mask = cv2.threshold(cropped_alpha, 30, 255, cv2.THRESH_BINARY)  # Higher threshold to remove borders
                
                # Additional morphological operations to remove thin borders
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=1)
                mask = cv2.dilate(mask, kernel, iterations=1)
            else:
                # For non-transparent images, use contour-based approach
                if channels == 4: char_img_bgr = cv2.cvtColor(char_img_cv, cv2.COLOR_BGRA2BGR)
                else: char_img_bgr = char_img_cv
                
                # Convert to grayscale and threshold
                char_gray = cv2.cvtColor(char_img_bgr, cv2.COLOR_BGR2GRAY)
                _, char_thresh = cv2.threshold(char_gray, 180, 255, cv2.THRESH_BINARY_INV)
                
                # Find inner contours (the actual character)
                inner_contours, _ = cv2.findContours(char_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Create mask from inner contours
                mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
                cv2.drawContours(mask, inner_contours, -1, (255), thickness=cv2.FILLED)

            # --- DEBUG VIS 4: Save the individual mask ---
            if mask is not None:
                cv2.imwrite(os.path.join(output_dir, f"_debug_mask_{i}.png"), mask)
            else:
                print(f"Warning: Mask for index {i} was not generated.")

            # Create a standardized output image (with transparency)
            std_img = np.zeros((max_height, max_width, 4), dtype=np.uint8)
            
            # Convert the cropped image to BGRA if needed
            if char_img_cv.shape[2] == 3:
                char_img_bgra = cv2.cvtColor(char_img_cv, cv2.COLOR_BGR2BGRA)
            elif char_img_cv.shape[2] == 4:
                char_img_bgra = char_img_cv.copy()
            else:
                print(f"Warning: Cropped image for index {i} has unexpected channels: {char_img_cv.shape}. Cannot process.")
                continue

            # Apply the mask to the alpha channel
            if mask is not None and mask.shape == char_img_bgra.shape[:2]:
                char_img_bgra[:, :, 3] = cv2.bitwise_and(char_img_bgra[:, :, 3] if char_img_bgra.shape[2] == 4 else np.full_like(mask, 255), mask)
            elif mask is not None:
                print(f"Warning: Mask shape {mask.shape} incompatible with alpha shape {char_img_bgra.shape[:2]} for index {i}.")

            # Calculate centering offsets
            x_offset = (max_width - crop_w) // 2
            y_offset = (max_height - crop_h) // 2

            # Place the character in the center of the standardized image
            std_img[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w] = char_img_bgra

            if characters and i < len(characters):
                char = characters[i]
                # if punctuation, use the name of the character
                if char in ".,?!":
                    char = {
                        '.': 'period',
                        ',': 'comma',
                        '?': 'question',
                        '!': 'exclamation'
                    }[char]
                output_filename = os.path.join(output_dir, f"{char}.png")
            else:
                output_filename = os.path.join(output_dir, f"{i}.png")

            try:
                success = cv2.imwrite(output_filename, std_img)
                if success:
                    char_label = f"'{characters[i]}'" if characters and i < len(characters) else str(i)
                    print(f"Saved: {output_filename} (Character {char_label}, Contour {i+1}/{len(valid_contours_boxes)})")
                else:
                    print(f"Error: Failed to save {output_filename} using cv2.imwrite.")
            except Exception as save_e:
                print(f"Error saving {output_filename} with cv2.imwrite: {save_e}")

        # (Final count checks - unchanged)
        if characters and num_found_chars < num_expected_chars:
             processed_count = i + 1 if 'i' in locals() else 0
             print(f"\nWarning: Processed {processed_count} contours, but expected {num_expected_chars} characters.")
        elif characters and num_found_chars > num_expected_chars:
             print(f"\nNote: Found {num_found_chars} contours initially, more than the {num_expected_chars} expected characters. Only the first {num_expected_chars} were processed based on limit.")

        print("\nProcessing complete.")

    except ImportError: print("Error: OpenCV, NumPy or Pillow not installed. Please install: pip install opencv-python numpy Pillow")
    except Exception as e: print(f"An unexpected error occurred: {e}"); traceback.print_exc()


if __name__ == "__main__":
    INPUT_IMAGE = sys.argv[1]
    OUTPUT_FOLDER = sys.argv[2]
    CHARACTERS = "abcdefghijklmnopqrstuvwxyz,.?!"
    PADDING = 5

    split_font_grid_contours_indexed(INPUT_IMAGE, OUTPUT_FOLDER, CHARACTERS, PADDING)
