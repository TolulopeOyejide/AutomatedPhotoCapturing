from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image = Image.open("/workspaces/AutomatedPhotoCapturing/data/lion-dry-forest-landscape_23-2151661802.jpg")

inputs = processor(image, return_tensors = "pt")

outputs = model.generate(**inputs)

caption = processor.decode(outputs[0],skip_special_tokens = True)

print("Generated Caption:", caption)