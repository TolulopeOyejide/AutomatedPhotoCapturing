import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load model and processor
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Get image URLs from a webpage
def extract_image_urls(url):
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

# Generate caption for an image URL
def generate_caption(img_url, processor, model):
    try:
        response = requests.get(img_url)
        raw_image = Image.open(BytesIO(response.content))

        if raw_image.size[0] * raw_image.size[1] < 400:
            return None

        raw_image = raw_image.convert("RGB")

        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption

    except Exception as e:
        print(f"Error processing image {img_url}: {e}")
        return None

# Main function to run the app
def run_image_captioning(url, output_file="caption.txt"):
    processor, model = load_model()
    image_urls = extract_image_urls(url)

    with open(output_file, "w") as file:
        for img_url in image_urls:
            caption = generate_caption(img_url, processor, model)
            if caption:
                file.write(f"{img_url}: {caption}\n")
                print(f"Captioned: {img_url}")

