import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from transformers import BlipProcessor, BlipForConditionalGeneration
import gradio as gr

# Load model and processor once
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def extract_valid_image_urls(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        img_elements = soup.find_all('img')

        valid_urls = []
        for img in img_elements:
            img_url = img.get('src')
            if not img_url:
                continue
            if 'svg' in img_url or '1x1' in img_url:
                continue
            if img_url.startswith('//'):
                img_url = 'https:' + img_url
            elif not img_url.startswith('http://') and not img_url.startswith('https://'):
                continue
            valid_urls.append(img_url)
        return valid_urls
    except Exception as e:
        return []

def generate_caption_from_url(img_url):
    try:
        response = requests.get(img_url)
        raw_image = Image.open(BytesIO(response.content))

        if raw_image.size[0] * raw_image.size[1] < 400:
            return None, None

        raw_image = raw_image.convert("RGB")
        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return raw_image, caption
    except Exception as e:
        return None, f"Error: {e}"

def caption_images_from_url(web_url):
    image_urls = extract_valid_image_urls(web_url)
    results = []

    for img_url in image_urls[:5]:  # limit to 5 for performance
        image, caption = generate_caption_from_url(img_url)
        if image and caption:
            results.append((image, caption))
    return results

# Gradio Interface
demo = gr.Interface(
    fn=caption_images_from_url,
    inputs=gr.Textbox(label="Enter Webpage URL"),
    outputs =gr.Gallery(label="Images with Captions"),
    title="🧠 AI-Powered Image Captioning from Webpage URL",
    description="Paste a webpage URL and get AI-generated captions for its images.",
    article= "<p style='text-align: center;'>Developed by Tolulope Oyejide </p>"
)

if __name__ == "__main__":
    demo.launch()
