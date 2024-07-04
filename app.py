import streamlit as st
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
import pytesseract

# Function to extract text using Tesseract OCR
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text

# Step 2: Upload the graph
st.title("Graph Interpreter with OCR")
uploaded_file = st.file_uploader("Upload a graph image", type=["png", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Graph.', use_column_width=True)
    
    # Convert the image to an array
    image_array = np.array(image)

    # Step 3: Read the graph using OpenCV and extract text
    st.write("Interpreting the graph...")
    
    # Extract text using OCR
    text_from_image = extract_text_from_image(image)
    st.write("Extracted Text from Image:")
    st.write(text_from_image)
    
    # Extract month names and other relevant text
    months = [month for month in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] if month in text_from_image]
    
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
        
        # Extract the points from the largest contour
        data_points = []
        for point in largest_contour:
            data_points.append(point[0])

        data_points = np.array(data_points)
        data_points = data_points[data_points[:, 0].argsort()]  # Sort by x-axis

        # Identify peaks and troughs
        peaks, _ = find_peaks(data_points[:, 1])
        troughs, _ = find_peaks(-data_points[:, 1])
        
        # Select a few significant peaks and troughs
        significant_peaks = data_points[peaks][:5] if len(peaks) > 5 else data_points[peaks]
        significant_troughs = data_points[troughs][:5] if len(troughs) > 5 else data_points[troughs]

        # Fit a trend line
        X = data_points[:, 0].reshape(-1, 1)
        y = data_points[:, 1]
        model = LinearRegression()
        model.fit(X, y)
        trend_line = model.predict(X)
        trend_description = "rising" if model.coef_[0] > 0 else "falling"

        # Calculate moving average
        window_size = 5
        moving_average = np.convolve(y, np.ones(window_size)/window_size, mode='valid')

        # Plot the annotated graph
        fig, ax = plt.subplots()
        ax.plot(data_points[:, 0], data_points[:, 1], label='Extracted Points')
        ax.plot(significant_peaks[:, 0], significant_peaks[:, 1], "x", label='Significant Peaks')
        ax.plot(significant_troughs[:, 0], significant_troughs[:, 1], "o", label='Significant Troughs')
        ax.plot(data_points[:, 0], trend_line, label='Trend Line', linestyle='--')
        ax.plot(data_points[window_size-1:, 0], moving_average, label='Moving Average', color='orange')
        ax.invert_yaxis()
        ax.legend()

        # Annotate peaks and troughs with corresponding month names
        if len(months) > 0:
            for i, point in enumerate(significant_peaks):
                ax.annotate(f'Peak\n{months[i%len(months)]}\n({point[0]}, {point[1]})', (point[0], point[1]), textcoords="offset points", xytext=(0,10), ha='center')
            for i, point in enumerate(significant_troughs):
                ax.annotate(f'Trough\n{months[i%len(months)]}\n({point[0]}, {point[1]})', (point[0], point[1]), textcoords="offset points", xytext=(0,-15), ha='center')
        
        st.pyplot(fig)

        # Display the summary in a text box
        summary = f"""
        **Graph Interpretation Summary**

        - **Trend**: The overall trend shows a {trend_description} pattern, indicating a {trend_description} in sales over the months.
        - **Peaks**: Significant peaks, indicating the highest sales, are observed in the months around {', '.join(months[:len(significant_peaks)])}.
        - **Troughs**: Significant troughs, indicating the lowest sales, are observed in the months around {', '.join(months[:len(significant_troughs)])}.
        - **Insights**: The moving average indicates a consistent upward trend, smoothing out short-term fluctuations.
        """
        st.text_area("Graph Interpretation Summary", summary, height=250)

    else:
        st.write("No contours found.")
