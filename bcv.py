import streamlit as st
import cv2
import numpy as np

def edge_detection(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    return edges

def corner_detection(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect corners using Shi-Tomasi method
    corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)

    # Convert corners to integer coordinates
    corners = np.int0(corners)

    # Draw circles at the corners
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(image, (x, y), 3, 255, -1)

    return image

def main():
    st.title("Edge and Corner Detection with Streamlit")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

        # Display the original image
        st.image(image, caption="Original Image", use_column_width=True)

        # Edge Detection
        edges = edge_detection(image)
        st.image(edges, caption="Edge Detection", use_column_width=True)

        # Corner Detection
        corners = corner_detection(image.copy())
        st.image(corners, caption="Corner Detection", use_column_width=True)

if __name__ == "__main__":
    main()
