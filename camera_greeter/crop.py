filename = "ziv.jpg"

maxwidth=384

from PIL import Image

def prepare_image_for_thermal_printer(filename, max_width=384):
    """
    Accepts a filename, scales the image, converts it to a dithered grayscale
    suitable for thermal printers, and saves the result.
    
    Args:
        filename (str): The path to the input image file.
        max_width (int): The maximum desired width for the output image.
                         Thermal printers often have a fixed width like 384 dots.
    
    Returns:
        str: The path to the generated output file.
    """
    print("Preparing ", filename, max_width )
    try:
        # Open the image file
        with Image.open(filename) as img:
            # 1. Scale the image to the maximum width
            original_width, original_height = img.size
            if original_width > max_width:
                scale_factor = max_width / original_width
                new_height = int(original_height * scale_factor)
                img = img.resize((max_width, new_height), Image.LANCZOS)
            
            # 2. Convert to dithered grayscale (1-bit mode)
            # The '1' mode in Pillow represents a 1-bit pixel (black and white).
            # The .convert() method with the '1' mode automatically applies a
            # dithering algorithm to simulate shades of gray.
            dithered_img = img.convert('1', dither=Image.Dither.FLOYDSTEINBERG)

            # 3. Generate the output filename and save the image
            output_filename = filename.rsplit('.', 1)[0] + '-snipped.png'
            dithered_img.save(output_filename)
            
            print(f"Image processed and saved to: {output_filename}")
            return output_filename
            
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# --- Example Usage ---
if __name__ == '__main__':
    # Replace 'your_image.jpg' with the path to your image file
    # Example:
    # prepare_image_for_thermal_printer('my_photo.jpg')
    
    # Or for a more dynamic use
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        prepare_image_for_thermal_printer(input_file)
    else:
        print("Please provide a filename as a command-line argument.")