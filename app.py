import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pytesseract
import plotly.graph_objs as go
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression

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
    
    # Ensure image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Convert the image to an array
    image_array = np.array(image)

    # Step 3: Read the graph using OpenCV and extract text
    st.write("Interpreting the graph...")
    
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

    # Extract data points (for simplicity, we'll assume the graph is a line chart)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
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

        # Plot the annotated graph using Plotly
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=data_points[:, 0], y=data_points[:, 1], mode='lines', name='Extracted Points'))
        fig.add_trace(go.Scatter(x=significant_peaks[:, 0], y=significant_peaks[:, 1], mode='markers+text', text=[f'Peak {months[i%len(months)]}' for i in range(len(significant_peaks))], textposition='top center', name='Significant Peaks', marker=dict(color='red')))
        fig.add_trace(go.Scatter(x=significant_troughs[:, 0], y=significant_troughs[:, 1], mode='markers+text', text=[f'Trough {months[i%len(months)]}' for i in range(len(significant_troughs))], textposition='bottom center', name='Significant Troughs', marker=dict(color='blue')))
        fig.add_trace(go.Scatter(x=data_points[:, 0], y=trend_line, mode='lines', name='Trend Line', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=data_points[window_size-1:, 0], y=moving_average, mode='lines', name='Moving Average', line=dict(color='orange')))

        fig.update_layout(title='Annotated Graph', xaxis_title='X-axis', yaxis_title='Y-axis', legend_title='Legend')
        fig.update_yaxes(autorange='reversed')

        st.plotly_chart(fig)

        # Generate peak and trough months
        peak_months = [months[i%len(months)] for i in range(len(significant_peaks))]
        trough_months = [months[i%len(months)] for i in range(len(significant_troughs))]

        # Display the summary in a markdown text box
        summary = f"""
### Graph Interpretation Summary

**Trend**: The overall trend shows a **{trend_description}** pattern, indicating a {trend_description} in sales over the months.

**Peaks**: Significant peaks, indicating the highest sales, are observed in the months around **{', '.join(peak_months)}**.

**Troughs**: Significant troughs, indicating the lowest sales, are observed in the months around **{', '.join(trough_months)}**.

**Insights**: The moving average indicates a consistent upward trend, smoothing out short-term fluctuations.
"""
        st.markdown(summary)

    else:
        st.write("No contours found.")
