# Import PIL (Python Imaging Library) for handling images
from PIL import Image

# Import PyTorch (required backend for Hugging Face models)
import torch

# Import Gradio for building a simple web interface
import gradio as gr

# Import BLIP model and processor (used for image captioning)
from transformers import BlipProcessor, BlipForConditionalGeneration

# Import dotenv to load environment variables (optional for API keys/configs)
from dotenv import load_dotenv

# Load environment variables from a .env file (if exists)
load_dotenv()


# -------------------------------------------------------
# Load BLIP (image captioning) model and processor
# -------------------------------------------------------

# Load the pre-trained BLIP model (for caption generation)
caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# Load the BLIP processor (for preparing images/text before feeding to model)
caption_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)


# -------------------------------------------------------
# Function to generate captions from an image
# -------------------------------------------------------

def generate_caption(image: Image.Image):
    """
    This function takes an image and generates a caption using BLIP.
    """
    # Preprocess the input image for the BLIP model
    inputs = caption_processor(images=image, return_tensors="pt")

    # Generate output (caption tokens) from the model
    output = caption_model.generate(**inputs)

    # Decode tokens into a human-readable caption string
    caption = caption_processor.decode(output[0], skip_special_tokens=True).strip()

    # Return the final caption text
    return caption


# -------------------------------------------------------
# Gradio User Interface (UI)
# -------------------------------------------------------

# Create a Gradio interface for the caption generator
interface = gr.Interface(
    fn=generate_caption,                      # Function to run when user uploads an image
    inputs=gr.Image(type="pil"),              # Input: Image (converted to PIL format)
    outputs=gr.Textbox(label="📝 Generated Caption"),  # Output: Textbox showing generated caption
    title="Never Give Up <br> 🖼️ AI Image Caption Generator",  # Title displayed on webpage
    description="Upload an image and let AI generate a smart caption using BLIP."  # Small description
)


# -------------------------------------------------------
# Run the app
# -------------------------------------------------------

# Launch the Gradio app only when this file is run directly
if __name__ == "__main__":
    interface.launch()