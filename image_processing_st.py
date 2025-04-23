from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import streamlit as st
import cv2
import matplotlib.pyplot as plt
import math
layout = [1,5,1]
def get_image_st(uploaded_file):
    col1, col2, col3 = st.columns(layout)
    with col2:
        st.subheader("1- Original and grayscale images")
        image_pil = Image.open(uploaded_file).convert('RGB')
        image = np.array(image_pil)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        with st.expander("Click to view"):
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            axs[0].imshow(image)
            axs[0].set_title("Original")
            axs[0].axis('off')

            axs[1].imshow(gray, cmap='gray')
            axs[1].set_title("Grayscale")
            axs[1].axis('off')

            st.pyplot(fig)

    return image_bgr, gray


def threshold_image_st(gray):
    col1, col2, col3 = st.columns(layout)
    with col2:
        st.subheader("2- Thresholding the image")
        with st.expander("Click to view"):
            st.info("Converts a grayscale image to a binary image using Otsu's method with inverse thresholding.")

            use_manual = st.checkbox("Enable manual thresholding below (not recommended for most cases)")
            if use_manual:
                manual_val = st.slider("Manual threshold value", 0, 255, 128)
                _, thresh = cv2.threshold(gray, manual_val, 255, cv2.THRESH_BINARY_INV)
            else:
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(thresh, cmap='gray')
            ax.set_title("Thresholded Image")
            ax.axis('off')
            st.pyplot(fig)

    return thresh

def remove_lines_st(image):
    col1, col2, col3 = st.columns(layout)
    with col2:
        st.subheader("3- Removing horizontal lines")
        with st.expander("Click to view"):
            st.info("Detects and removes horizontal lines from a binary image.  \nUseful for cleaning scanned images with notebook or grid lines.") 

            kernel_width = st.slider("Horizontal kernel width", min_value=10, max_value=150, value=40, step=2)
            kernel_size = (kernel_width, 1)

            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
            detected_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

            line_mask = cv2.bitwise_not(detected_lines)
            cleaned = cv2.bitwise_and(image, image, mask=line_mask)

            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            axs[0].imshow(detected_lines, cmap='gray')
            axs[0].set_title("Detected Lines")
            axs[0].axis('off')

            axs[1].imshow(cleaned, cmap='gray')
            axs[1].set_title("Cleaned Image")
            axs[1].axis('off')

            st.pyplot(fig)

    return cleaned


def fig_to_array(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(int(height), int(width), 4)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return rgb_image

def get_digits_batches_st(original_image, image):
    col1, col2, col3 = st.columns(layout)
    with col2:
        st.subheader("4- Extracting digits from image")

        with st.expander("Click to view"):
            st.info("Detects digit regions (batches) in the image using contour detection.  \nIgnores regions whose height is below a threshold fraction of the tallest contour (to eliminate small noisy components).")
            height_thresh = st.slider("Minimum height percentage of tallest digit", 0.1, 1.0, 0.5)

            contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            max_height = 0
            for cnt in contours:
                _, _, _, h = cv2.boundingRect(cnt)
                max_height = max(max_height, h)

            digits = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if h < height_thresh * max_height:
                    continue
                digit = image[y:y + h, x:x + w]
                digits.append(((x, y, w, h), digit))

            digits.sort(key=lambda item: item[0][0])
            digit_images = [item[1] for item in digits]

            contour_img = original_image.copy()
            for (x, y, w, h), _ in digits:
                cv2.rectangle(contour_img, (x, y), (x + w, y + h), (0, 255, 0), 3)

            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
            ax1.set_title("Detected Digit Regions")
            ax1.axis('off')

            img1 = fig_to_array(fig1)
            plt.close(fig1)

            cols = 4
            rows = math.ceil(len(digit_images) / cols)
            fig2, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
            axs = axs.flatten()

            for i, digit_img in enumerate(digit_images):
                axs[i].imshow(digit_img, cmap='gray')
                axs[i].set_title(f"Digit {i+1}")
                axs[i].axis('off')

            for i in range(len(digit_images), len(axs)):
                axs[i].axis('off')

            fig2.tight_layout()
            img2 = fig_to_array(fig2)
            plt.close(fig2)

            h = max(img1.shape[0], img2.shape[0])
            img1_padded = np.pad(img1, ((0, h - img1.shape[0]), (0, 0), (0, 0)), constant_values=255)
            img2_padded = np.pad(img2, ((0, h - img2.shape[0]), (0, 0), (0, 0)), constant_values=255)
            merged = np.hstack((img1_padded, img2_padded))

            fig, ax = plt.subplots(figsize=(14, h / 100))
            ax.imshow(merged)
            ax.axis('off')
            st.pyplot(fig)

    return digit_images


def process_digits_st(digits_batches):
    col1, col2, col3 = st.columns(layout)
    with col2:
        st.subheader("5- Resizing, padding, and normalizing digits")

        with st.expander("Click to view"):
            st.info("Processes a list of digit images by resizing, padding, and normalizing them for model input.  \nEach digit is resized to a height of $24$ pixels, padded to $28 \\times 28$, and normalized to $[0, 1]$.  \nOptionally thickens the digits using morphological dilation.")
            thicken = st.checkbox("Thicken digits", value=True)
            processed_digits = []

            for digit in digits_batches:
                h, w = digit.shape[:2]

                scale = 24 / h
                new_w = int(w * scale)
                new_w = min(new_w, 24)
                resized = cv2.resize(digit, (new_w, 24), interpolation=cv2.INTER_AREA)

                total_pad = 28 - new_w
                pad_left = total_pad // 2
                pad_right = total_pad - pad_left

                padded = cv2.copyMakeBorder(resized, 2, 2, pad_left, pad_right,
                                            cv2.BORDER_CONSTANT, value=0)
                processed_digits.append(padded)

            if thicken:
                kernel = np.ones((2, 2), np.uint8)
                processed_digits = [cv2.dilate(digit, kernel, iterations=1) for digit in processed_digits]

            processed_digits = [(digit / 255.0).reshape((28, 28, 1)) for digit in processed_digits]
            processed_digits = np.array(processed_digits)

            cols = 4
            rows = max(math.ceil(len(processed_digits) / cols),1)
            fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
            axs = axs.flatten()

            for i, digit in enumerate(processed_digits):
                axs[i].imshow(digit.squeeze(), cmap='gray')
                axs[i].set_title(f"Digit {i+1}")
                axs[i].axis('off')

            for i in range(len(processed_digits), len(axs)):
                axs[i].axis('off')

            fig.suptitle("Final 28x28 Processed Digits", fontsize=16)
            fig.tight_layout()
            st.pyplot(fig)

    return processed_digits


def predict_st(processed_digits, model, image):
    col1, col2, col3 = st.columns(layout)
    with col2:
        st.subheader("6- Predicted sequence")

        y_pred = model.predict(processed_digits, verbose=0)
        predicted_string = ''.join(str(np.argmax(y)) for y in y_pred)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(image)
        ax.set_title(f'Predicted Number: {predicted_string}', fontsize=16)
        ax.axis('off')
        st.pyplot(fig)
    return predicted_string

