# Image Caption and Search Application
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from nltk.corpus import stopwords
from textblob import Word
import json
import os
from datetime import datetime, timedelta
from collections import Counter
import time
import concurrent.futures
import threading
st.set_page_config(layout="wide")

stop_words = set(stopwords.words('english'))

THUMBNAIL_SIZE = (400, 400)
THUMBNAIL_DIR = "Thumbnails"
MAX_WORKERS = 3

@st.cache_resource
def load_captioning_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

# loading the model
processor, model = load_captioning_model()

thread_local = threading.local()

# for tab buttons
st.markdown("""<style>
    .keywords-section {
        border: 1px solid #e6e6e6;
        border-radius: 5px;
        padding: 10px;
        background-color: #f8f9fa;
        height: calc(100vh - 130px);
        overflow-y: auto;
        margin-top: 0;
    }
    
    .stDataFrame {
        height: 100%;
    }
    
    [data-testid="column"] > div:first-child {
        margin-top: 0;
        padding-top: 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 4px 4px 0 0;
        padding: 0px 16px;
        font-weight: 500;
        color: black;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }
</style>""", unsafe_allow_html=True)

def save_dict_to_file(dictionary, filename):
    with open(filename, 'w') as f:
        json.dump(dictionary, f)


def load_dict_from_file(filename):
    try:
        with open(filename, 'r') as f:
            content = f.read().strip()
            if content:
                return json.loads(content)
            else:
                return {}
    except (FileNotFoundError, json.JSONDecodeError):
        save_dict_to_file({}, filename)
        return {}

if not os.path.exists('database.txt'):
    save_dict_to_file({}, 'database.txt')

def resize_image_for_processing(image):
    image = image.convert('RGB')
    image.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
    return image

# Core AI function: Generates descriptive caption for the given image
def image_caption(image):
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_length=500)
    return processor.decode(out[0], skip_special_tokens=True)

# Processes uploaded image: saves original, creates thumbnail, generates AI caption, and extracts keywords
def process_image(img_data):
    uploaded_img, unique_id = img_data
    try:
        image = Image.open(uploaded_img)
        img_name = uploaded_img.name
        unique_filename = f"{unique_id}.{img_name.split('.')[-1]}"
        
        original_path = os.path.join('Images', unique_filename)
        image.save(original_path)
        
        thumbnail = resize_image_for_processing(image)
        thumbnail_path = os.path.join(THUMBNAIL_DIR, unique_filename)
        thumbnail.save(thumbnail_path)
        
        text = image_caption(thumbnail)
        text_removest = [word for word in text.split() if word not in stop_words]
        text_singular = [Word(word).singularize() for word in text_removest]
        
        return {
            'success': True,
            'filename': unique_filename,
            'original_name': img_name,
            'keywords': text_singular
        }
    except Exception as e:
        return {
            'success': False,
            'filename': uploaded_img.name,
            'error': str(e)
        }

# Searches database for images that match all provided search terms
def find_keys_by_words(dictionary, input_words):
    input_words = input_words.split()
    input_words = [Word(word).singularize().lower() for word in input_words]
    matching_keys = []
    for key, value in dictionary.items():
        value_lower = [w.lower() for w in value]
        if all(any(word in v.lower() for v in value_lower) for word in input_words):
            matching_keys.append(key)
    return matching_keys

def get_terms_for_table(dictionary):
    word_counts = Counter()
    for words_list in dictionary.values():
        for word in words_list:
            word_counts[word.lower()] += 1
    
    terms_data = [{"Keyword": word, "Images": count} for word, count in sorted(word_counts.items())]
    return terms_data

loaded_dict = load_dict_from_file('database.txt')

st.title("Image Caption and Search App")
tab1, tab2 = st.tabs(["📤 Image Upload", "🔍 Image Search"])

with tab1:
    st.header('Upload Images')
    uploaded_imgs = st.file_uploader("Upload images:", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        upload_button = st.button("Upload Images", use_container_width=True)
        
    if upload_button and uploaded_imgs:
        start_time = time.time()
        progress_text = "Uploading and processing images..."
        my_bar = st.progress(0, text=progress_text)
        status_placeholder = st.empty()
        results_placeholder = st.empty()
        
        os.makedirs('Images', exist_ok=True)
        os.makedirs(THUMBNAIL_DIR, exist_ok=True)
        
        loaded_dict = load_dict_from_file('database.txt')
        
        total_images = len(uploaded_imgs)
        image_tasks = [(img, f"{img.name.split('.')[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}") 
                      for i, img in enumerate(uploaded_imgs)]
        processed_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_img = {executor.submit(process_image, img_data): img_data for img_data in image_tasks}
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_img)):
                result = future.result()
                processed_results.append(result)
                
                progress = (i + 1) / total_images
                my_bar.progress(progress, text=f"Processing images... {i+1}/{total_images}")
                
                if result['success']:
                    loaded_dict[result['filename']] = result['keywords']
                    status_placeholder.info(f"Processing image {i+1}/{total_images}. Processed {i+1} of {total_images} images.")
                    
                    results_area = "<div style='margin-top: 10px;'>"
                    for res in processed_results:
                        if res['success']:
                            keywords_str = ", ".join(res['keywords']) if 'keywords' in res else ""
                            results_area += f"✅ {res['original_name']}: {keywords_str}<br>"
                        else:
                            results_area += f"❌ {res['filename']}: Error - {res['error']}<br>"
                    results_area += "</div>"
                    results_placeholder.markdown(results_area, unsafe_allow_html=True)
                else:
                    st.error(f"Failed to process {result['filename']}: {result['error']}")
        
        save_dict_to_file(loaded_dict, 'database.txt')
        
        elapsed = time.time() - start_time
        if elapsed > 60:
            time_msg = f"Completed in {int(elapsed // 60)} min {int(elapsed % 60)} sec"
        else:
            time_msg = f"Completed in {int(elapsed)} seconds"
            
        status_placeholder.success(f"✅ All {total_images} images processed in parallel. {time_msg}")

with tab2:
    st.header('Search Images by Text')
    
    if not loaded_dict:
        st.warning("📸 The database is empty. Please upload some images using the Image Upload tab first.")
    else:
        search_col, keywords_col = st.columns([2, 1])
        
        with search_col:
            user_input = st.text_input('Enter search terms (separate multiple words with spaces):')
            search_button = st.button('Search Images', use_container_width=True)
            
            if search_button:
                if not user_input.strip():
                    st.warning("Please enter search terms to find images.")
                else:
                    result = find_keys_by_words(loaded_dict, user_input)
                    if result:
                        st.write(f"Found {len(result)} matching images:")
                        
                        img_cols = st.columns(3)
                        for i, img_name in enumerate(result):
                            img_path = os.path.join('Images', img_name)
                            if os.path.exists(img_path):
                                with img_cols[i % 3]:
                                    img = Image.open(img_path)
                                    st.image(img, caption=f"Image {i+1}", use_container_width=True)
                    else:
                        st.info("No matching images found. Try different search terms.")
        
        with keywords_col:
            st.subheader("Available Keywords")
                        
            terms_data = get_terms_for_table(loaded_dict)
            if terms_data:
                import pandas as pd
                df = pd.DataFrame(terms_data)
                st.dataframe(df, use_container_width=True, height=600)
            else:
                st.info("No keywords available yet")
            
            st.markdown("</div>", unsafe_allow_html=True)
