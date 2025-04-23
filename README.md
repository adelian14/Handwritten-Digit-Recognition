# ✍️ Handwritten Digit Recognition App

An interactive web app for recognizing handwritten digits from images, built using Python, Streamlit, OpenCV, and a CNN model trained on MNIST.

This project isn't meant to be perfect or production-grade — it's built to be **explorable**, **educational**, and a little fun.

---

## 🎯 What Does This App Do?

You upload an image of handwritten digits — like something written in a notebook or on a piece of paper — and the app walks you through each step of the digit recognition process:

1. **Convert the image to grayscale**
2. **Apply thresholding** (Otsu or manual)
3. **Remove horizontal notebook lines**
4. **Detect and crop individual digits**
5. **Resize, pad, and normalize each digit**
6. **Pass the digits to a CNN model**
7. **Display the final prediction**

At each step, you can tweak the settings and immediately see how it affects the output.

---

## ⚙️ Technologies Used

- **Python** for all logic and processing  
- **Streamlit** for building the interactive web app  
- **OpenCV** for image preprocessing  
- **Matplotlib** for visual previews  
- **TensorFlow / Keras** to load and train the digit recognition model (CNN on MNIST)

---

## 🚀 Try It Out

🔗 **Live Demo:** [https://handwritten-digit-recognition-922.streamlit.app/](https://handwritten-digit-recognition-922.streamlit.app/)

---

## 📄 Function Reference

### `get_image_st(uploaded_file)`
Loads the image and converts it to grayscale.  
Returns both color and grayscale versions.

---

### `threshold_image_st(gray)`
Applies binary inverse thresholding (Otsu or manual).  
Returns the thresholded image.

---

### `remove_lines_st(image)`
Removes horizontal lines using morphological operations.  
Useful for cleaning notebook backgrounds.

---

### `get_digits_batches_st(original_image, image)`
Finds contours, crops individual digits, and returns them sorted left-to-right.

---

### `process_digits_st(digits_batches)`
Resizes digits to 28x28, normalizes them, and optionally thickens them.

---

### `get_model(model_path)`
Loads a CNN model trained on MNIST. Trains and saves it if not already available.

---

### `predict_st(processed_digits, model, image)`
Predicts each digit and displays the final result with the original image.

---

## 🙏 Why I Built This

I built this as a fun way to explore image processing + ML in an interactive format.  
The goal wasn't perfection — but to learn, share, and maybe help others visualize the process too.

---

## 💡 Future Ideas

- Applying more advanced image processing techniques  
- Support drawing digits inside the app  
- Compare different model architectures

---

## 👨‍💻 Author

Made with curiosity, learning, and a little love.  
If you find this helpful or interesting, feel free to reach out or fork the repo!

