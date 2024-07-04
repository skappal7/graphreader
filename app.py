import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pytesseract
import pandas as pd
from datetime import datetime
import re

def detect_chart_type(image):
    # This is a placeholder function. In a real implementation, 
    # you'd use more sophisticated image processing or machine learning here.
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    
    if lines is not None and len(lines) > 10:
        return "line"
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 10:
        return "bar"
    
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    if circles is not None:
        return "pie"
    
    return "scatter"

def extract_text_and_data(image, chart_type):
    text = pytesseract.image_to_string(image)
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    if chart_type == "line":
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            data_points = np.array([point[0] for point in largest_contour])
            data_points = data_points[data_points[:, 0].argsort()]
    elif chart_type == "bar":
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        data_points = np.array([(cnt[0][0][0], gray.shape[0] - cnt.max(axis=0)[0][1]) for cnt in contours])
        data_points = data_points[data_points[:, 0].argsort()]
    elif chart_type == "pie":
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            center = circles[0, 0]
            data_points = []
            for i in range(0, 360, 10):
                x = int(center[0] + 0.9 * center[2] * np.cos(np.radians(i)))
                y = int(center[1] + 0.9 * center[2] * np.sin(np.radians(i)))
                if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                    data_points.append((i, gray[y, x]))
            data_points = np.array(data_points)
    else:  # scatter
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        data_points = np.array([cnt.mean(axis=0)[0] for cnt in contours if len(cnt) == 1])
    
    return text, data_points

def generate_insights(data_points, chart_type):
    insights = []
    
    if chart_type == "line" or chart_type == "bar":
        min_val = data_points[:, 1].min()
        max_val = data_points[:, 1].max()
        avg_val = data_points[:, 1].mean()
        insights.append((f"The minimum value is {min_val:.2f}", ["minimum"]))
        insights.append((f"The maximum value is {max_val:.2f}", ["maximum"]))
        insights.append((f"The average value is {avg_val:.2f}", ["average"]))
        
        if chart_type == "line":
            trend = "upward" if data_points[-1, 1] > data_points[0, 1] else "downward"
            insights.append((f"The overall trend is {trend}", [trend]))
        
    elif chart_type == "pie":
        total = data_points[:, 1].sum()
        max_slice = data_points[data_points[:, 1].argmax()]
        min_slice = data_points[data_points[:, 1].argmin()]
        insights.append((f"The largest slice is at {max_slice[0]} degrees, representing {(max_slice[1]/total)*100:.2f}% of the total", ["largest"]))
        insights.append((f"The smallest slice is at {min_slice[0]} degrees, representing {(min_slice[1]/total)*100:.2f}% of the total", ["smallest"]))
        
    else:  # scatter
        x_min, x_max = data_points[:, 0].min(), data_points[:, 0].max()
        y_min, y_max = data_points[:, 1].min(), data_points[:, 1].max()
        insights.append((f"The x-axis ranges from {x_min:.2f} to {x_max:.2f}", ["x-axis"]))
        insights.append((f"The y-axis ranges from {y_min:.2f} to {y_max:.2f}", ["y-axis"]))
        
    return insights

def color_text(text, words_to_color, color):
    for word in words_to_color:
        text = text.replace(word, f'<span style="color:{color}">{word}</span>')
    return text

st.title("Multi-Chart Interpreter with OCR")

st.sidebar.header("Customize Colors")
positive_color = st.sidebar.color_picker("Pick a color for positive trends", "#FF0000")
negative_color = st.sidebar.color_picker("Pick a color for negative trends", "#00FF00")

show_table = st.sidebar.checkbox("Show data table", False)

uploaded_file = st.file_uploader("Upload a chart image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Chart', use_column_width=True)
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    chart_type = detect_chart_type(image)
    st.write(f"Detected chart type: {chart_type}")
    
    text, data_points = extract_text_and_data(image, chart_type)
    
    if data_points is not None:
        insights = generate_insights(data_points, chart_type)
        
        st.write("### Chart Interpretation Summary")
        for insight, words_to_color in insights:
            colored_text = color_text(insight, words_to_color, positive_color if any(word in ["upward", "maximum", "largest"] for word in words_to_color) else negative_color)
            st.markdown(colored_text, unsafe_allow_html=True)
        
        if show_table:
            st.write("### Extracted Data")
            df = pd.DataFrame(data_points, columns=['X', 'Y'])
            st.dataframe(df)
    else:
        st.write("Could not extract data points from the image. Please check the image quality.")
else:
    st.write("Please upload an image to begin analysis.")
