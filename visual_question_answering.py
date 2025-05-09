import requests
from PIL import Image

from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
raw_image = Image.open(requests.get(img_url, stream = True).raw).convert('RGB')

question = "What is in the image?"

inputs = processor(raw_image, question, return_tensors = "pt")

out = model.generate(**inputs)

answer = processor.decode(out[0], skip_special_tokens = True)

print(f"Answer: {answer}")