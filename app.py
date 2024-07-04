import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pytesseract
import pandas as pd
from scipy.signal import find_peaks

def extract_text_and_data(image):
    # Extract text using OCR
    text = pytesseract.image_to_string(image)
    
    # Convert image to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Use Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (assuming it's the graph line)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Extract data points
        data_points = np.array([point[0] for point in largest_contour])
        
        # Sort data points by x-coordinate
        data_points = data_points[data_points[:, 0].argsort()]
        
        return text, data_points
    
    return text, None

st.title("Graph Interpreter with OCR")

uploaded_file = st.file_uploader("Upload a graph image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Graph', use_column_width=True)
    
    # Ensure image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Extract text and data points
    text, data_points = extract_text_and_data(image)
    
    if data_points is not None:
        # Extract month names
        months = [month for month in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] if month in text]
        
        # Identify peaks and troughs
        peaks, _ = find_peaks(data_points[:, 1])
        troughs, _ = find_peaks(-data_points[:, 1])
        
        # Prepare data for DataFrame
        peak_months = [months[i % len(months)] for i in range(len(peaks))]
        trough_months = [months[i % len(months)] for i in range(len(troughs))]
        peak_values = data_points[peaks][:, 1]
        trough_values = data_points[troughs][:, 1]
        
        df_data = {
            "Month": peak_months + trough_months,
            "Value": list(peak_values) + list(trough_values),
            "Type": ["High"] * len(peaks) + ["Low"] * len(troughs)
        }
        
        df = pd.DataFrame(df_data)
        
        # Color customization
        st.sidebar.header("Customize Colors")
        high_color = st.sidebar.color_picker("High value color", "#00FF00")
        low_color = st.sidebar.color_picker("Low value color", "#FF0000")
        
        # Apply conditional formatting
        def color_cells(val):
            color = high_color if val == "High" else low_color
            return f'background-color: {color}'
        
        styled_df = df.style.applymap(color_cells, subset=['Type'])
        
        # Display table
        st.write("### High and Low Values")
        st.dataframe(styled_df)
        
        # Generate and display summary
        window_size = min(3, len(data_points))
        trend = 'rising' if np.mean(data_points[:, 1]) < np.mean(data_points[-window_size:, 1]) else 'falling'
        
        summary = f"""
        ### Graph Interpretation Summary
        
        **Trend**: The overall trend shows a {trend} pattern.
        
        **Peaks**: Highest values observed in {', '.join(peak_months)}.
        
        **Troughs**: Lowest values observed in {', '.join(trough_months)}.
        
        **Insight**: The data suggests {trend} values over time with fluctuations.
        """
        
        st.markdown(summary)
    else:
        st.write("Could not extract data points from the image. Please try a different image.")
else:
    st.write("Please upload an image to begin analysis.")
