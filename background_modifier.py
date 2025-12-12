import torch
import numpy as np
from PIL import Image, ImageFilter
import os
import glob
import urllib.request
from tqdm import tqdm
from ultralytics import YOLOWorld
from segment_anything import sam_model_registry, SamPredictor

INPUT_ROOT = "./input_images"
OUTPUT_ROOT = "./output_images"

DEVICE_YOLO = "cpu"
DEVICE_SAM = "cuda" if torch.cuda.is_available() else "cpu"

SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"

BLUR_RADIUS = 15       # Lager it gets more blur
PIXEL_SIZE = 20        # Larger it gets more pixelized

# downloading model
def download_sam_weights():
    if not os.path.exists(SAM_CHECKPOINT):
        print("Downloading SAM model")
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        urllib.request.urlretrieve(url, SAM_CHECKPOINT)
        print("Download Complete")

download_sam_weights()

try:
    detector = YOLOWorld('yolov8l-worldv2.pt')
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE_SAM)
    predictor = SamPredictor(sam)
    print("Model loaded")
except Exception as e:
    print(f"Failed initialization: {e}")
    exit()

def get_yolo_box(image_path):
    detector.set_classes(["car"])
    results = detector.predict(image_path, conf=0.1, verbose=False, device="cpu")
    if len(results[0].boxes) == 0: return None

    best_box = None
    max_area = 0
    for box in results[0].boxes.xyxy.numpy():
        xmin, ymin, xmax, ymax = box
        area = (xmax - xmin) * (ymax - ymin)
        if area > max_area:
            max_area = area
            best_box = box 
    return best_box

def process_and_save(image_path, relative_path):
    try:
        # 1. image load
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        H, W = image_np.shape[:2]

        # 2. YOLO
        detected_box = get_yolo_box(image_path)
        if detected_box is None: return

        # 3. SAM
        predictor.set_image(image_np)
        masks, _, _ = predictor.predict(
            point_coords=None, point_labels=None,
            box=detected_box[None, :], multimask_output=False
        )
        mask_bool = masks[0] 
        mask_uint8 = (mask_bool * 255).astype(np.uint8)

        # 4. segmanting
        mask_image = Image.fromarray(mask_uint8, mode="L")
        original_rgba = image_pil.convert("RGBA")
        
        car_transparent = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        car_transparent.paste(original_rgba, (0, 0), mask=mask_image)
        
        rows, cols = np.where(mask_bool)
        if len(rows) == 0: return
        
        # original car location
        y_min, y_max = rows.min(), rows.max()
        x_min, x_max = cols.min(), cols.max()
        
        # cropping car
        car_crop = car_transparent.crop((x_min, y_min, x_max, y_max))



        dirs = {
            "noise": os.path.join(OUTPUT_ROOT, "noise", relative_path),
            "white": os.path.join(OUTPUT_ROOT, "white", relative_path),
            "blur":  os.path.join(OUTPUT_ROOT, "blur", relative_path),
            "pixel": os.path.join(OUTPUT_ROOT, "pixel", relative_path),
        }
        

        for path in dirs.values():
            os.makedirs(os.path.dirname(path), exist_ok=True)

        # ---------------------------------------------------------
        # [Case 1] Noise Background
        random_bg_array = np.random.randint(0, 256, size=(H, W, 3), dtype=np.uint8)
        bg_noise = Image.fromarray(random_bg_array, mode="RGB").convert("RGBA")
        bg_noise.paste(car_crop, (x_min, y_min), car_crop)
        bg_noise.convert("RGB").save(dirs["noise"])

        # [Case 2] White Background
        bg_white = Image.new("RGBA", (W, H), (255, 255, 255, 255))
        bg_white.paste(car_crop, (x_min, y_min), car_crop)
        bg_white.convert("RGB").save(dirs["white"])

        # [Case 3] Blur Background 
        bg_blur = original_rgba.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
        bg_blur.paste(car_crop, (x_min, y_min), car_crop)
        bg_blur.convert("RGB").save(dirs["blur"])

        # [Case 4] Pixelate Background 
        small_w = max(1, W // PIXEL_SIZE)
        small_h = max(1, H // PIXEL_SIZE)
        bg_pixel = original_rgba.resize((small_w, small_h), resample=Image.BILINEAR)
        bg_pixel = bg_pixel.resize((W, H), resample=Image.NEAREST)
        
        bg_pixel.paste(car_crop, (x_min, y_min), car_crop)
        bg_pixel.convert("RGB").save(dirs["pixel"])
        # ---------------------------------------------------------

    except Exception as e:
        print(f"Error ({os.path.basename(image_path)}): {e}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_ROOT):
        print(f"No {INPUT_ROOT} folder.")
        exit()

    all_files = []
    for root, dirs, files in os.walk(INPUT_ROOT):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, INPUT_ROOT)
                all_files.append((full_path, rel_path))

    print(f" image count : {len(all_files)}")
    print(f"  noise, white, blur, pixel")
    
    for full_p, rel_p in tqdm(all_files):
        process_and_save(full_p, rel_p)
        
    print("Image process done")