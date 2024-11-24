# Import libraries
import streamlit as st
from PIL import Image
import openai
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import time

# Set up OpenAI API key
openai.api_key = "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"

# Load the BLIP model and processor once with caching
@st.cache_resource
def load_model():
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    return model, processor

model, processor = load_model()

# Function to generate captions from images using BLIP
def generate_captions_from_images(images):
    captions = []
    for image in images:
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        captions.append(caption)
    return captions

# Function to generate a story from captions using GPT-3.5
def generate_story_from_captions(captions):
    prompt = "Create a story based on these captions: " + " ".join(captions)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a creative storyteller."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7,
    )
    story = response['choices'][0]['message']['content']
    return story

# Streamlit UI Setup
st.set_page_config(page_title="Interactive AI Storyteller", page_icon="📚", layout="wide")

st.title("📚 Interactive AI Storyteller")
st.write("Create a unique story based on captions generated from your uploaded images.")

# Styling for instructions and captions
st.markdown("""
<style>
    .instructions { font-size: 1.1rem; color: #4B4B4B; }
    .story-text { font-size: 1.2rem; line-height: 1.6; color: #333333; font-family: 'Georgia', serif; }
    .caption-text { font-style: italic; color: #666666; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="instructions">Upload at least 4 images below, and our AI will craft a captivating story just for you.</div>', unsafe_allow_html=True)

# File uploader for images
uploaded_files = st.file_uploader("Choose at least 4 images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

# Ensure minimum number of images are uploaded
if uploaded_files and len(uploaded_files) >= 4:
    st.write("### Uploaded Images")
    images = [Image.open(file) for file in uploaded_files]
    st.image(images, caption=[file.name for file in uploaded_files], width=150)

    # Generate captions and story on button click
    if st.button("Generate Story"):
        with st.spinner("Generating captions..."):
            captions = generate_captions_from_images(images)
            time.sleep(1)

        # Display captions
        st.write("### Generated Captions")
        for i, caption in enumerate(captions):
            st.markdown(f'<div class="caption-text">Image {i + 1}: {caption}</div>', unsafe_allow_html=True)
            time.sleep(0.5)

        # Display generated story
        with st.spinner("Creating your story..."):
            story = generate_story_from_captions(captions)
            time.sleep(1)

        st.write("### Your AI-Generated Story")
        st.markdown(f'<div class="story-text">{story}</div>', unsafe_allow_html=True)
else:
    st.write("Please upload at least 4 images to continue.")
