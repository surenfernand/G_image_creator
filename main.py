import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler
import numpy as np

# Load pre-trained model and scheduler
model_id = "stabilityai/stable-diffusion-2-1"
scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    torch_dtype=torch.float32
).to("cuda")

# Disable safety checker for artistic flexibility
pipe.run_safety_checker = lambda images, device, dtype: (images, [False] * len(images))

# Load your real image (resize it to around 512x512 for best results)
input_image_path = "real_image.jpg"
image = Image.open(input_image_path).convert("RGB").resize((1080, 1080))

# Prompt tuned for Ghibli style
prompt = (
    "A realistic Studio Ghibli style illustration, ultra high definition, sharp focus, crisp edges, "
    "highly detailed, intricate textures, cinematic lighting, anime scenery, vivid colors, "
    "fantasy atmosphere, painterly yet realistic style, natural environment"
)

# Generate Ghibli-styled image
generated_image = pipe(
    prompt=prompt,
    image=image,
    strength=0.8,  # Controls how much to alter the original image (0 = minimal, 1 = strong)
    guidance_scale=10,
    num_inference_steps=20
).images[0]

# Save output
result_np = np.array(generated_image)
result_np = np.nan_to_num(result_np, nan=0.0, posinf=255.0, neginf=0.0).astype(np.uint8)
Image.fromarray(result_np).save("ghibli_style_output.png")
print("✅ Ghibli-style image saved as ghibli_style_output.png")
