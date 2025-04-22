import streamlit as st
from image_processing_st import get_image_st, threshold_image_st, remove_lines_st, get_digits_batches_st, process_digits_st, predict_st
from model import get_model

import streamlit.components.v1 as components

st.set_page_config(layout="wide")

col1, col2, col3 = st.columns([1,6,1])
with col2:
    st.title("✍️ Handwritten Digit Recognition")

    st.markdown("""
    👋 **Welcome!** This is an interactive app that detects and recognizes handwritten digits from uploaded images.  
    🧾 Whether your digits are written on paper, in a notebook, or scanned documents, this tool processes them step by step to extract and predict each digit with high accuracy.

    ---

    ### ⚙️ **How it works:**
    - 🤖 Uses a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**, containing 60,000 handwritten digits.
    - 🧪 Your image goes through a pipeline: grayscale conversion, thresholding, noise removal, digit segmentation, and preprocessing — all customizable by you.
    - 👁️ You can preview each stage, tweak the settings, and see how it affects the final prediction.

    📤 **Upload a clear image of digits and follow along the steps below!**
    """)



    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image, gray = get_image_st(uploaded_file)
    thresh = threshold_image_st(gray)
    cleaned = remove_lines_st(thresh)
    digits_batches = get_digits_batches_st(image, cleaned)
    processed_digits = process_digits_st(digits_batches)
    model = get_model('digit_model.h5')
    prediction = predict_st(processed_digits, model, image)
    col1, col2, col3 = st.columns([1,4,1])
    with col2:
        st.markdown(
            "<h3 style='font-size: 20px; font-weight: 600;'>Is the prediction correct? If not, try adjusting the options in the steps above.</h3>",
            unsafe_allow_html=True
        )
