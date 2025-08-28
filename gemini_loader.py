from __future__ import annotations
import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import hashlib
import os

import folder_paths
from comfy.comfy_types.node_typing import IO
from comfy_api_nodes.apis import (
    GeminiContent,
    GeminiGenerateContentRequest,
    GeminiGenerateContentResponse,
    GeminiInlineData,
    GeminiPart,
    GeminiMimeType,
)
from comfy_api_nodes.apis.client import (
    SynchronousOperation,
)
from comfy_api_nodes.apinode_utils import (
    tensor_to_base64_string,
)
from comfy_api_nodes.nodes_gemini import GeminiModel, get_gemini_endpoint
from server import PromptServer

# --- Instruction Sets ---
NSFW_INSTRUCTIONS = {
    "PONY": """Your task is to generate a prompt that will recreate the provided image using the PonyXL model. The prompt must be a comma-separated list of Danbooru-style tags.\n\nFollow these rules strictly:\n1.  **Tagging:** Use descriptive, comma-separated Danbooru tags.\n2.  **Content:** Include tags for score (e.g., `score_9_up`, `score_8_up`), explicit content, and exaggerated physical features like `large breasts`, `huge ass`, `giant penis`. Be specific and detailed.\n3.  **Format:** Your entire output must be ONLY the prompt itself. Do not include any other text, headers, labels, explanations, or negative prompts. Your response should begin with the first tag and end with the last tag.""",
    "FLUX": """Craft a Flux.1-style realistic prompt to recreate the provided image. Use flowing, natural language with comma-separated descriptive tags, focusing on hyper-exaggerated body proportions like (massively oversized breasts:1.3), (enormous buttocks:1.4), and (gigantic penis:1.5) with explicit, vivid detail. Start with the main subject and scene, weave in medium, style, colors, lighting, and finish with quality enhancers like \"hyper-realistic, ultra-detailed, 8k, cinematic\". Output only the prompt itself, excluding headers, labels, negative prompts, or any additional text, ensuring it aligns with Flux.1’s advanced natural language processing for stunning photorealistic generation.""",
    "1.5": """Craft a Stable Diffusion 1.5-style realistic prompt to recreate the provided image. Use comma-separated descriptive tags, apply emphasis weights (e.g., (keyword:1.2)) on key features, and explicitly exaggerate body proportions with hyper-detailed, explicit language for elements like massively oversized breasts, enormous buttocks, and gigantic penis. Start with the main subject and scene for optimal guidance, include quality enhancers like \"highly detailed, realistic, 8k\" at the end. Output only the prompt itself, excluding headers, labels, negative prompts, or any additional text, ensuring it aligns with SD 1.5's natural language style for photorealistic generation.""",
    "SDXL": """Craft a Stable Diffusion XL (SDXL)-style realistic prompt to recreate the provided image. Use detailed natural language with comma-separated descriptive tags, apply emphasis weights (e.g., (keyword:1.2)) on key elements for stronger impact, and explicitly exaggerate body proportions with vivid, hyper-detailed explicit language like (enormous breasts:1.3), (massive buttocks:1.4), and (gigantic penis:1.5). Structure the prompt iteratively: begin with the main subject and scene, layer in medium, style, additional details, colors, lighting, and end with quality boosters like \"photorealistic, highly detailed, 8k, masterpiece\". Output only the prompt itself, excluding headers, labels, negative prompts, or any additional text, ensuring it aligns with SDXL's advanced natural language handling for superior photorealistic generation."""
}

SFW_INSTRUCTIONS = {
    "PONY": """Your task is to generate a prompt that will recreate the provided image using the PonyXL model. The prompt must be a comma-separated list of Danbooru-style tags.\n\nFollow these rules strictly:\n1.  **Tagging:** Use descriptive, comma-separated Danbooru tags for characters, objects, and scenery.\n2.  **Content:** Include tags for score (e.g., `score_9_up`, `score_8_up`). Do not include any explicit or NSFW content.\n3.  **Format:** Your entire output must be ONLY the prompt itself. Do not include any other text, headers, labels, explanations, or negative prompts. Your response should begin with the first tag and end with the last tag.""",
    "FLUX": """Craft a Flux.1-style realistic prompt to recreate the provided image. Use flowing, natural language with comma-separated descriptive tags. Start with the main subject and scene, weave in medium, style, colors, lighting, and finish with quality enhancers like 'hyper-realistic, ultra-detailed, 8k, cinematic'. Output only the prompt itself, excluding headers, labels, negative prompts, or any additional text.""",
    "1.5": """Craft a Stable Diffusion 1.5-style realistic prompt to recreate the provided image. Use comma-separated descriptive tags. Start with the main subject and scene for optimal guidance, include quality enhancers like 'highly detailed, realistic, 8k' at the end. Output only the prompt itself, excluding headers, labels, negative prompts, or any additional text.""",
    "SDXL": """Craft a Stable Diffusion XL (SDXL)-style realistic prompt to recreate the provided image. Use detailed natural language with comma-separated descriptive tags. Structure the prompt iteratively: begin with the main subject and scene, layer in medium, style, additional details, colors, lighting, and end with quality boosters like 'photorealistic, highly detailed, 8k, masterpiece'. Output only the prompt itself, excluding headers, labels, negative prompts, or any additional text."""
}

# Helper functions from comfyui-fitsize
resample_filters = {
    'nearest': 0,
    'lanczos': 1,
    'bilinear': 2,
    'bicubic': 3,
}

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def get_image_size(image: torch.Tensor) -> tuple[int, int]:
    samples = image.movedim(-1, 1)
    size = samples.shape[3], samples.shape[2]
    return size

def octal_sizes(width, height):
    octalwidth = width if width % 8 == 0 else width + (8 - width % 8)
    octalheight = height if height % 8 == 0 else height + (8 - height % 8)
    return (octalwidth, octalheight)

def get_max_size(width, height, max_size, upscale="false"):
    aspect_ratio = width / height

    fit_width = max_size
    fit_height = max_size

    if upscale == "false" and width <= max_size and height <= max_size:
        return (width, height, aspect_ratio)
    
    if aspect_ratio > 1:
        fit_height = int(max_size / aspect_ratio)
    else:
        fit_width = int(max_size * aspect_ratio)

    new_width, new_height = octal_sizes(fit_width, fit_height)

    return (new_width, new_height, aspect_ratio)

class GeminiImageLoader:
    """
    A node that loads, resizes, and analyzes an image to generate a prompt using the Gemini API,
    and provides a preview of the final image.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        image_files = folder_paths.filter_files_content_types(files, ["image"])
        
        prompt_styles = ["PONY", "FLUX", "1.5", "SDXL"]

        return {
            "required": {
                "image": (sorted(image_files), {"image_upload": True}),
                "model": (
                    [model.value for model in GeminiModel],
                    {
                        "tooltip": "The Gemini model to use for generating the prompt.",
                        "default": GeminiModel.gemini_2_5_pro_preview_05_06.value,
                    },
                ),
                "prompt_style": (prompt_styles, {"default": "PONY", "tooltip": "The style of prompt to generate."}),
                "nsfw": ("BOOLEAN", {"default": True, "label_on": "NSFW", "label_off": "SFW"}),
                "max_size": ("INT", {"default": 1024, "step": 8, "tooltip": "The maximum dimension (width or height) for the image."}),
                "upscale": (["false", "true"], {"tooltip": "Whether to upscale images smaller than max_size."}),
                "resampling": (list(resample_filters.keys()), {"default": "lanczos", "tooltip": "The resampling method to use for resizing."}),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Seed for the generation. Deterministic output is not guaranteed.",
                    },
                ),
            },
            "optional": {
                "vae": ("VAE",)
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "LATENT")
    RETURN_NAMES = ("image", "mask", "prompt", "latent")
    FUNCTION = "load_and_generate_prompt"
    CATEGORY = "Gemini"

    def create_image_parts(self, image_input: torch.Tensor) -> list[GeminiPart]:
        image_parts: list[GeminiPart] = []
        for image_index in range(image_input.shape[0]):
            image_as_b64 = tensor_to_base64_string(
                image_input[image_index].unsqueeze(0)
            )
            image_parts.append(
                GeminiPart(
                    inlineData=GeminiInlineData(
                        mimeType=GeminiMimeType.image_png,
                        data=image_as_b64,
                    )
                )
            )
        return image_parts

    def create_text_part(self, text: str) -> GeminiPart:
        return GeminiPart(text=text)

    def get_text_from_response(self, response: GeminiGenerateContentResponse) -> str:
        if not response.candidates:
            return "No response candidates found."
        parts = response.candidates[0].content.parts
        return "\n".join([part.text for part in parts if hasattr(part, 'text') and part.text])

    async def load_and_generate_prompt(
        self,
        image: str,
        model: str,
        prompt_style: str,
        nsfw: bool,
        max_size: int,
        upscale: str,
        resampling: str,
        seed: int,
        unique_id: str | None = None,
        vae = None,
        **kwargs,
    ) -> dict:
        # 1. Load the image from file
        image_path = folder_paths.get_annotated_filepath(image)
        pil_img = Image.open(image_path)
        
        output_images, output_masks = [], []
        for i in ImageSequence.Iterator(pil_img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image_tensor_part = i.convert("RGB")
            image_tensor_part = np.array(image_tensor_part).astype(np.float32) / 255.0
            image_tensor_part = torch.from_numpy(image_tensor_part)[None,]
            
            if 'A' in i.getbands():
                mask_part = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask_part = 1. - torch.from_numpy(mask_part)
            else:
                mask_part = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            
            output_images.append(image_tensor_part)
            output_masks.append(mask_part.unsqueeze(0))

        if len(output_images) > 1:
            image_tensor = torch.cat(output_images, dim=0)
            mask_tensor = torch.cat(output_masks, dim=0)
        else:
            image_tensor = output_images[0]
            mask_tensor = output_masks[0]

        # 2. Resize the image
        original_size = get_image_size(image_tensor)
        new_width, new_height, _ = get_max_size(original_size[0], original_size[1], max_size, upscale)
        
        img_to_resize = tensor2pil(image_tensor)
        resized_img = img_to_resize.resize((new_width, new_height), resample=Image.Resampling(resample_filters[resampling]))
        resized_tensor = pil2tensor(resized_img)

        # 3. Create image preview of the *resized* image
        results = []
        (full_output_folder, filename, counter, subfolder, _) = folder_paths.get_save_image_path("GeminiLoader", self.output_dir)
        file = f"{filename}_{counter:05}_.png"
        resized_img.save(os.path.join(full_output_folder, file))
        results.append({"filename": file, "subfolder": subfolder, "type": self.type})

        # 4. Select instructions and generate prompt
        instruction_map = NSFW_INSTRUCTIONS if nsfw else SFW_INSTRUCTIONS
        instructions = instruction_map.get(prompt_style, instruction_map["PONY"]) # Default to PONY style

        parts: list[GeminiPart] = [self.create_text_part(instructions)]
        image_parts = self.create_image_parts(resized_tensor)
        parts.extend(image_parts)

        gemini_model_enum = GeminiModel(model)

        response = await SynchronousOperation(
            endpoint=get_gemini_endpoint(gemini_model_enum),
            request=GeminiGenerateContentRequest(
                contents=[
                    GeminiContent(
                        role="user",
                        parts=parts,
                    )
                ]
            ),
            auth_kwargs=kwargs,
        ).execute()

        output_text = self.get_text_from_response(response)
        if unique_id and output_text:
            PromptServer.instance.send_progress_text(output_text, node_id=unique_id)
        
        generated_prompt = output_text or "Empty response from Gemini model..."

        # 5. Encode to latent if VAE is provided
        if vae:
            latent = vae.encode(resized_tensor[:,:,:,:3])
            latent_output = {"samples": latent}
        else:
            # Create an empty latent
            latent_output = {"samples": torch.zeros([1, 4, new_height // 8, new_width // 8])}

        return {"ui": {"images": results}, "result": (resized_tensor, mask_tensor, generated_prompt, latent_output)}

    @classmethod
    def IS_CHANGED(cls, image: str, model: str, prompt_style: str, nsfw: bool, max_size: int, upscale: str, resampling: str, vae=None, **kwargs):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        # Also consider other params in cache invalidation
        m.update(model.encode())
        m.update(prompt_style.encode())
        m.update(str(nsfw).encode())
        m.update(str(max_size).encode())
        m.update(upscale.encode())
        m.update(resampling.encode())
        if vae:
            m.update(vae.name.encode())
        return m.digest().hex()

NODE_CLASS_MAPPINGS = {
    "GeminiImageLoader": GeminiImageLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImageLoader": "Gemini Image Loader & Prompter"
}
