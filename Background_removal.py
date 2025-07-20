from rembg import remove # type: ignore
from PIL import Image
import io

def remove_background(input_image_path, output_image_path):
    with open(input_image_path, "rb") as input_file:
        input_image = input_file.read()
        output_image = remove(input_image)
    result = Image.open(io.BytesIO(output_image))
    result.save(output_image_path)
    
    print(f"Background removed successfully! Image saved to {output_image_path}")

if __name__ == "__main__":
    input_path = "person3.png"  
    output_path = "output3.png"
    remove_background(input_path, output_path)
