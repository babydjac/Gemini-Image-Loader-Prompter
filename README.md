
# ♊️ Gemini Image Loader & Prompter for ComfyUI

This custom node for ComfyUI is a powerful, all-in-one tool designed to streamline your image-to-prompt workflow. It seamlessly integrates image loading, resizing, and intelligent prompt generation using Google's Gemini API, giving you a versatile starting point for your creations.

Load an image, and instantly get a resized and VAE-encoded version, plus a highly customizable, AI-generated prompt in various styles to match your target model.

## ✨ Features

- **All-in-One Workflow**: Combines Image Loading, Resizing, Prompt Generation, and VAE Encoding into a single, convenient node.
- **Intelligent Prompt Generation**: Leverages the Gemini API to analyze an image and generate high-quality prompts.
- **Multiple Prompt Styles**: Includes a dropdown to generate prompts specifically tailored for **PonyXL**, **Flux**, **Stable Diffusion 1.5**, and **SDXL**.
- **NSFW Toggle**: Easily switch between generating SFW and NSFW content with a dedicated toggle.
- **Advanced Resizing**: Integrates the functionality of `comfyui-fitsize` to resize images to a maximum dimension while preserving aspect ratio, with multiple resampling methods.
- **Optional VAE Encoding**: Connect a VAE to automatically encode the resized image into a latent representation.
- **Live Preview**: Shows a preview of the final resized image directly in the node.

## 📸 Examples

Here are examples of prompts generated for different models:

*Example for Stable Diffusion 1.5:*
![Example for Stable Diffusion 1.5](1.5Ex.jpeg)

*Example for Flux:*
![Example for Flux](FluxEx.jpeg)

*Example for Pony:*
![Example for Pony](PonyEx.jpeg)

*Example for SDXL:*
![Example for SDXL](SDXLEx.jpeg)

## ⚙️ Installation

1.  Navigate to your ComfyUI `custom_nodes` directory.
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  Clone this repository:
    ```bash
    git clone https://github.com/babydjac/Gemini-Image-Loader-Prompter.git
    ```
3.  Restart ComfyUI.

## 🚀 Usage

1.  **Add the Node**: Right-click on the canvas, select "Add Node", and go to `Gemini > Gemini Image Loader & Prompter`.
2.  **Select an Image**: Use the `image` dropdown to select an image from your `ComfyUI/input` directory, or use the "Choose File to Upload" button.
3.  **Choose a Prompt Style**: Select your target style (PONY, FLUX, 1.5, SDXL) from the `prompt_style` dropdown.
4.  **Toggle NSFW**: Enable or disable NSFW content generation.
5.  **Set Resize Options**: Adjust the `max_size`, `upscale`, and `resampling` parameters to control how the image is resized.
6.  **Connect a VAE (Optional)**: If you want a latent output, connect a VAE to the `vae` input.
7.  **Queue Prompt**: The node will output the resized image, a mask, the generated prompt, and an optional latent.

## ⚠️ Dependencies

This node relies on the `comfy_api_nodes` infrastructure, which should be part of your existing ComfyUI environment. Ensure it is properly installed.

## 🔑 API Key

This node is secure and does **not** store your API key. It uses the authentication system provided by `comfy_api_nodes`. Your Gemini API key should be handled by that system.
