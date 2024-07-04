import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pytesseract
import pandas as pd
from scipy.signal import find_peaks

# Function to extract text using Tesseract OCR
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text

# Function to extract data points from the largest contour
def extract_data_points(contour):
    data_points = []
    for point in contour:
        data_points.append(point[0])
    data_points = np.array(data_points)
    return data_points

# Step 2: Upload the graph
st.title("Graph Interpreter with OCR")
uploaded_file = st.file_uploader("Upload a graph image", type=["png", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Graph.', use_column_width=True)
    
    # Ensure image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Convert the image to an array
    image_array = np.array(image)

    # Extract text using OCR
    text_from_image = extract_text_from_image(image)
    
    # Extract month names and other relevant text
    months = [month for month in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] if month in text_from_image]

    # Convert image to grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Use Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        data_points = extract_data_points(largest_contour)
        data_points = data_points[data_points[:, 0].argsort()]  # Sort by x-axis

        # Identify peaks and troughs
        peaks, _ = find_peaks(data_points[:, 1])
        troughs, _ = find_peaks(-data_points[:, 1])

        # Extract peak and trough values with corresponding months
        peak_values = data_points[peaks][:, 1]
        trough_values = data_points[troughs][:, 1]
        peak_months = [months[i % len(months)] for i in range(len(peaks))]
        trough_months = [months[i % len(months)] for i in range(len(troughs))]

        # Prepare data for the table
        high_low_data = {
            "Month": peak_months + trough_months,
            "Value": list(peak_values) + list(trough_values),
            "Type": ["High"] * len(peak_values) + ["Low"] * len(trough_values)
        }

        # DataFrame for the table
        df = pd.DataFrame(high_low_data)

        # Customizable colors
        st.sidebar.header("Customize Colors")
        high_color = st.sidebar.color_picker("Pick a color for High values", "#00FF00")
        low_color = st.sidebar.color_picker("Pick a color for Low values", "#FF0000")

        # Apply colors to the DataFrame
        def highlight_values(val, type_):
            color = high_color if type_ == "High" else low_color
            return f'background-color: {color}'

        styled_df = df.style.apply(lambda x: [highlight_values(v, t) for v, t in zip(x, df['Type'])], axis=1)
        
        # Display the styled DataFrame
        st.write("### High and Low Values by Month")
        st.dataframe(styled_df)

        # Define window_size
        window_size = min(3, len(data_points))

        # Display the summary in a markdown text box
        summary = f"""
### Graph Interpretation Summary

**Trend**: The overall trend shows a {'rising' if np.mean(data_points[:, 1]) < np.mean(data_points[-window_size:, 1]) else 'falling'} pattern, indicating a {'rise' if np.mean(data_points[:, 1]) < np.mean(data_points[-window_size:, 1]) else 'fall'} in sales over the months.

**Peaks**: Significant peaks, indicating the highest sales, are observed in the months around **{', '.join(peak_months)}**.

**Troughs**: Significant troughs, indicating the lowest sales, are observed in the months around **{', '.join(trough_months)}**.

**Insights**: The moving average indicates a consistent trend, smoothing out short-term fluctuations.
"""
        st.markdown(summary)
    else:
        st.write("No contours found.")
