from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from rembg import remove
from skimage import measure
from skimage.transform import resize
import trimesh
import os

app = FastAPI()
os.makedirs("outputs", exist_ok=True)

@app.get("/")
def root():
    return {"message": "CartoonPrint API is live."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate-stl/")
async def generate_stl(file: UploadFile = File(...)):
    try:
        input_image = Image.open(file.file).convert("RGBA")

        # Resize image if too large
        max_dim = 512
        if input_image.width > max_dim or input_image.height > max_dim:
            input_image.thumbnail((max_dim, max_dim))

        # Enhance contrast
        enhanced_image = ImageEnhance.Contrast(input_image).enhance(1.8)

        # Background removal
        output_image = remove(enhanced_image)
        img_np = np.array(output_image)

        # Convert to sketch
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)

        # Extract contours
        small = resize(sketch / 255.0, (256, 256))
        binary = small < 0.5
        contours = measure.find_contours(binary, 0.5)

        meshes = []
        height = 10.0
        for contour in contours:
            points_2d = contour[:, [1, 0]]
            points_3d = np.column_stack([points_2d, np.zeros_like(points_2d[:, 0])])
            top = np.column_stack([points_2d, np.full_like(points_2d[:, 0], height)])
            vertices = np.vstack([points_3d, top])
            faces = []
            n = len(points_2d)
            for i in range(n):
                j = (i + 1) % n
                faces += [[i, j, n + j], [i, n + j, n + i]]
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            meshes.append(mesh)

        final_mesh = trimesh.util.concatenate(meshes)
        base_name = file.filename.split('.')[0]
        stl_path = f"outputs/{base_name}.stl"
        final_mesh.export(stl_path)

        scene = final_mesh.scene()
        scene.camera.center = final_mesh.centroid
        scene.set_camera(angles=(np.radians(90), 0, 0), distance=200)
        preview_path = f"outputs/{base_name}_preview.png"
        png = scene.save_image(resolution=(512, 512), visible=True)
        with open(preview_path, "wb") as f:
            f.write(png)

        return {
            "stl_url": f"/{stl_path}",
            "preview_url": f"/{preview_path}"
        }
    except Exception as e:
        return {"error": str(e)}