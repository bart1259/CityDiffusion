import fastapi
from fastapi.middleware.cors import CORSMiddleware
import diffusers
import argparse
import base64
from PIL import Image
import io
import torch
import numpy as np

noise_scheduler = None
model = None
device = "cuda"

app = fastapi.FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TILE_SIZE = 256

# Host static files for testing
@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse("citygen.html")

# Post procesisng because sometimes images come out skewed
def post_process_image(input_image, include_alpha=False):
    from sklearn.cluster import KMeans
    
    pixel_values = np.array(np.array(input_image).reshape((TILE_SIZE*TILE_SIZE,3 if not include_alpha else 4)))
    kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto') # n_init='auto' is the recommended setting
    
    kmeans.fit(pixel_values)
    labels = kmeans.predict(pixel_values)
    
    PURE_RED = np.array([[255,0,0]]) if not include_alpha else np.array([[255,0,0,255]])
    PURE_BLUE = np.array([[0,0,255]]) if not include_alpha else np.array([[0,0,255,255]])
    
    red_index = ((kmeans.cluster_centers_ - PURE_RED) ** 2).sum(axis=-1).argmin()
    blue_index = ((kmeans.cluster_centers_ - PURE_BLUE) ** 2).sum(axis=-1).argmin()
    
    if red_index == blue_index:
        blue_index = np.argsort(((kmeans.cluster_centers_ - PURE_RED) ** 2).sum(axis=-1))[1]
    
    white_index = -1
    if (red_index == 0 or blue_index ==0) and (red_index == 1 or blue_index == 1):
        white_index = 2
    elif (red_index == 0 or blue_index ==0) and (red_index == 2 or blue_index == 2):
        white_index = 1
    elif (red_index == 2 or blue_index ==2) and (red_index == 1 or blue_index == 1):
        white_index = 0
    else:
        printf("WTF")
    
    pixel_labels = labels.reshape(TILE_SIZE,TILE_SIZE)
    processed_img = np.zeros_like(input_image, dtype=np.uint8)
    processed_img[pixel_labels == white_index] = [255,255,255] if not include_alpha else [255,255,255,255]
    processed_img[pixel_labels == blue_index] = [0,0,255] if not include_alpha else [0,0,255,255]
    processed_img[pixel_labels == red_index] = [255,0,0] if not include_alpha else [255,0,0,255]

    return processed_img

@app.post("/generate")
async def run_model(body: dict):
    image_b64 = body.get("image_b64")
    mask_b64 = body.get("mask_b64")
    num_inference_steps = body.get("num_inference_steps", 50)
    num_inference_steps = int(num_inference_steps)

    # 1. convert b64 strings to images
    image_data = base64.b64decode(image_b64)
    image_bytes = io.BytesIO(image_data)
    init_image = Image.open(image_bytes).convert("RGB")

    mask_data = base64.b64decode(mask_b64)
    mask_bytes = io.BytesIO(mask_data)
    mask_image = Image.open(mask_bytes).convert("RGB")

    input_image = np.asarray(init_image)
    mask_image = np.asarray(mask_image)

    print(f"Input image shape: {input_image.shape}, Mask image shape: {mask_image.shape}")

    noise_scheduler.set_timesteps(num_inference_steps)

    with torch.no_grad():
        input_image = torch.from_numpy(input_image).permute(2,0,1).unsqueeze(0).to(device).float() / 255.0
        mask_image  = torch.from_numpy(mask_image).permute(2,0,1).unsqueeze(0).to(device).float() / 255.0

        current_image = input_image.clone()

        for t in noise_scheduler.timesteps:  # <- use scheduler timesteps, already on CUDA
            print(f"Processing timestep {t}")
            # noise = torch.randn_like(current_image)
            # noisy_image = noise_scheduler.add_noise(current_image, noise, t)

            model_output = model(current_image, t).sample
            current_image = noise_scheduler.step(model_output, t, current_image).prev_sample

            # Apply mask
            current_image[0][mask_image[0,:,:] > 0.5] = input_image[0][mask_image[0,:,:] > 0.5]

        # 3. convert output tensor to image
        current_image = (current_image / 2 + 0.5).clamp(0, 1)
        output_image = (current_image.squeeze(0).permute(1,2,0).clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
        output_image = post_process_image(output_image, include_alpha=False)
        output_pil = Image.fromarray(output_image)
        buffered = io.BytesIO()
        output_pil.save(buffered, format="PNG")
        output_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {
            "image": output_b64
        }

# parse arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000, help='Port to run the API server on')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the diffusion model')
    args = parser.parse_args()

    # load model
    print(f"Loading model from {args.model_path}...")
    pipe = diffusers.DDPMPipeline.from_pretrained(args.model_path)
    pipe.to(device)
    print("Model loaded.")

    noise_scheduler = pipe.scheduler
    model = pipe.unet

    # run API server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)