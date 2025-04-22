import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

def get_image(image_path, show = True):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if show:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title('Original')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Grayscale')
        plt.imshow(gray, cmap='gray')
        plt.axis('off')

        plt.show()
    return image, gray

def threshold_image(image, show = True):
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if show:
        plt.figure(figsize=(8, 4))
        plt.title('Thresholded (Binary Inverse)')
        plt.imshow(thresh , cmap = 'gray')
        plt.axis('off')
        plt.show()
        
    return thresh

def remove_lines(image, kernel_size = (40, 1), show = True):
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    detected_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    line_mask = cv2.bitwise_not(detected_lines)

    cleaned = cv2.bitwise_and(image, image, mask=line_mask)
    
    if show:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title("Detected Horizontal Lines")
        plt.imshow(detected_lines, cmap='gray')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title("Thresholded Image After Removing Lines")
        plt.imshow(cleaned, cmap='gray')
        plt.axis('off')
        plt.show()
        
    return cleaned

def get_digits_batches(original_image, image, highet_thresh = 0.5, show = True):
    contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_height = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        max_height = max(max_height, h)
    
    digits = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if h < highet_thresh*max_height:
            continue
        digit = image[y:y + h, x:x + w]

        digits.append(((x, y, w, h),digit))
    digits.sort(key=lambda item : item[0][0])
    digit_images = [item[1] for item in digits]
    
    
    
    if show:
        contour_img = original_image.copy()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h < highet_thresh*max_height:
                continue
            cv2.rectangle(contour_img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        plt.figure(figsize=(10, 4))
        plt.title('Detected Digits')
        plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        
        cols = 4
        rows = math.ceil(len(digit_images) / cols)

        plt.figure(figsize=(cols*2, rows * 2))

        for i, digit in enumerate(digit_images):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(digit, cmap='gray')
            plt.title(f'Digit {i + 1}')
            plt.axis('off')

        plt.suptitle("Cropped Individual Digits", fontsize=16)
        plt.tight_layout()
        plt.show()
        
    return digit_images
    
def process_digits(digits_batches, thicken = False, show = True):
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
    processed_digits = [(digit/255).reshape((28,28,1)) for digit in processed_digits]
    processed_digits = np.array(processed_digits)
    
    if show:
        cols = 4
        rows = math.ceil(len(processed_digits) / cols)

        plt.figure(figsize=(cols * 2, rows * 2))

        for i, digit in enumerate(processed_digits):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(digit, cmap='gray')
            plt.axis('off')
            plt.title(f'Digit {i+1}')

        plt.suptitle("Final 28x28 Digits", fontsize=16)
        plt.tight_layout()
        plt.show()
    
    return processed_digits

def predict(processed_digits, model, image):
    y_pred = model.predict(processed_digits, verbose = 0)
    predicted_string = ''
    for y in y_pred:
        predicted_string = predicted_string + str(np.argmax(y))
    plt.figure(figsize=(8, 4))
    plt.title(f'Predicted number: {predicted_string}')
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    return predicted_string