import streamlit as st
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Step 2: Upload the graph
st.title("Graph Interpreter")
uploaded_file = st.file_uploader("Upload a graph image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Graph.', use_column_width=True)
    
    # Convert the image to an array
    image_array = np.array(image)

    # Step 3: Read the graph using OpenCV
    st.write("Interpreting the graph...")

    # Convert image to grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # Use Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    image_with_contours = image_array.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 3)
    
    # Display the contours
    st.image(image_with_contours, caption='Contours Highlighted.', use_column_width=True)

    # Step 4: Interpret the graph
    st.write("Graph interpretation highlights:")

    # Extract data points (for simplicity, we'll assume the graph is a line chart)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Highlight the bounding box
        image_with_box = image_array.copy()
        cv2.rectangle(image_with_box, (x, y), (x+w, y+h), (255, 0, 0), 2)
        st.image(image_with_box, caption='Largest Contour Highlighted.', use_column_width=True)
        
        st.write(f"Bounding box of the largest contour: x={x}, y={y}, width={w}, height={h}")

        # Extract the points from the largest contour
        data_points = []
        for point in largest_contour:
            data_points.append(point[0])

        data_points = np.array(data_points)
        data_points = data_points[data_points[:, 0].argsort()]  # Sort by x-axis

        # Plot the extracted points
        fig, ax = plt.subplots()
        ax.plot(data_points[:, 0], data_points[:, 1], label='Extracted Points')
        ax.invert_yaxis()
        st.pyplot(fig)

        # Identify peaks
        peaks, _ = find_peaks(data_points[:, 1])
        st.write(f"Identified peaks at x-coordinates: {data_points[peaks, 0]}")

        # Plot with peaks
        fig, ax = plt.subplots()
        ax.plot(data_points[:, 0], data_points[:, 1], label='Extracted Points')
        ax.plot(data_points[peaks, 0], data_points[peaks, 1], "x", label='Peaks')
        ax.invert_yaxis()
        ax.legend()
        st.pyplot(fig)

    else:
        st.write("No contours found.")
