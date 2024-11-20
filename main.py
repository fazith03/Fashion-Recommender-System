import streamlit as st
import tensorflow
from PIL import Image
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import os

# Load pre-trained data
features_list = pickle.load(open("image_features_embedding.pkl", "rb"))
img_files_list = pickle.load(open("img_files.pkl", "rb"))

# Model initialization
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

# App Title and Introduction
st.title('🛍️ Fashion Recommender System')
st.markdown(
    """
    **Welcome!**  
    Upload an image of your fashion item, and we’ll recommend similar items for you!  
    Explore, rate, and even download your recommendations. 🎯  
    """
)

# Save uploaded file
def save_file(uploaded_file):
    try:
        with open(os.path.join("uploader", uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
            return 1
    except Exception as e:
        st.error(f"File save error: {e}")
        return 0

# Feature extraction
def extract_img_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img)
    flatten_result = result_to_resnet.flatten()
    result_normlized = flatten_result / norm(flatten_result)
    return result_normlized

# Recommendation
def recommend(features, features_list, num_recommendations=5):
    neighbors = NearestNeighbors(n_neighbors=num_recommendations + 1, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)
    distances, indices = neighbors.kneighbors([features])
    return indices[0][1:]  # Exclude the uploaded image itself

# File uploader
uploaded_file = st.file_uploader("Upload your fashion image 📸", type=["png", "jpg", "jpeg"])
num_recommendations = st.slider("Select the number of recommendations:", min_value=1, max_value=10, value=5)

if uploaded_file is not None:
    if save_file(uploaded_file):
        st.success("Image uploaded successfully!")
        
        # Display uploaded image
        uploaded_img = Image.open(uploaded_file)
        st.image(uploaded_img, caption="Uploaded Image", use_column_width=True)

        with st.spinner("✨ Analyzing your image and finding similar fashion items..."):
            # Extract features
            features = extract_img_features(os.path.join("uploader", uploaded_file.name), model)
            recommended_indices = recommend(features, features_list, num_recommendations=num_recommendations)
        
        st.markdown("### Your Recommendations:")
        for idx in recommended_indices:
            col1, col2 = st.columns(2)

            # Display the recommended image
            with col1:
                st.image(img_files_list[idx], caption="Recommended Item", use_column_width=True)

            # Interactivity for the recommendation
            with col2:
                st.button(f"❤️ Save Item {idx}", key=f"save_{idx}")
                st.download_button(
                    label="📥 Download Image",
                    data=open(img_files_list[idx], "rb").read(),
                    file_name=f"recommendation_{idx}.jpg",
                    mime="image/jpeg",
                )
                st.radio(
                    "How do you feel about this recommendation?",
                    ("😍 Love it!", "🙂 It's okay", "🙁 Not for me"),
                    key=f"feedback_{idx}",
                )
    else:
        st.error("Failed to upload the file. Please try again.")
else:
    st.info("Please upload an image to get started.")
