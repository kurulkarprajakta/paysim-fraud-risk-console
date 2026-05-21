import json
import base64
import os

notebook = "Prajakta_Kurulkar_HW1.ipynb"
output_dir = "extracted_images"

os.makedirs(output_dir, exist_ok=True)

with open(notebook, "r", encoding="utf-8") as f:
    data = json.load(f)

count = 0

for cell in data["cells"]:
    if "outputs" in cell:
        for output in cell["outputs"]:
            if "data" in output and "image/png" in output["data"]:
                img = base64.b64decode(output["data"]["image/png"])
                filename = f"{output_dir}/image_{count}.png"
                with open(filename, "wb") as img_file:
                    img_file.write(img)
                count += 1

print(f"Extracted {count} images to {output_dir}")