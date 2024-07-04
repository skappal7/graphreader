import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pytesseract
import pandas as pd
from scipy.signal import find_peaks

def extract_text_and_data(image):
    text = pytesseract.image_to_string(image)
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        data_points = np.array([point[0] for point in largest_contour])
        data_points = data_points[data_points[:, 0].argsort()]
        return text, data_points
    
    return text, None

def generate_insights(df):
    insights = []
    
    # Overall trend
    first_value = df['Value'].iloc[0]
    last_value = df['Value'].iloc[-1]
    percent_change = ((last_value - first_value) / first_value) * 100
    trend = "increasing" if percent_change > 0 else "decreasing"
    insights.append(f"The overall trend is {trend} with a {abs(percent_change):.2f}% change from {df['Month'].iloc[0]} to {df['Month'].iloc[-1]}.")

    # Highest and lowest values
    max_row = df.loc[df['Value'].idxmax()]
    min_row = df.loc[df['Value'].idxmin()]
    insights.append(f"The highest value was {max_row['Value']:.2f} in {max_row['Month']}, while the lowest was {min_row['Value']:.2f} in {min_row['Month']}.")

    # Comparison to average
    avg_value = df['Value'].mean()
    above_avg = df[df['Value'] > avg_value]
    below_avg = df[df['Value'] < avg_value]
    insights.append(f"The average value across all months was {avg_value:.2f}.")
    insights.append(f"{len(above_avg)} months were above average: {', '.join(above_avg['Month'])}.")
    insights.append(f"{len(below_avg)} months were below average: {', '.join(below_avg['Month'])}.")

    # Month-to-month changes
    df['Change'] = df['Value'].pct_change() * 100
    max_increase = df.iloc[df['Change'].idxmax()]
    max_decrease = df.iloc[df['Change'].idxmin()]
    insights.append(f"The largest month-to-month increase was {max_increase['Change']:.2f}% from {df.iloc[df['Change'].idxmax() - 1]['Month']} to {max_increase['Month']}.")
    insights.append(f"The largest month-to-month decrease was {abs(max_decrease['Change']):.2f}% from {df.iloc[df['Change'].idxmin() - 1]['Month']} to {max_decrease['Month']}.")

    return insights

st.title("Graph Interpreter with OCR")

uploaded_file = st.file_uploader("Upload a graph image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Graph', use_column_width=True)
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    text, data_points = extract_text_and_data(image)
    
    if data_points is not None:
        months = [month for month in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] if month in text]
        
        peaks, _ = find_peaks(data_points[:, 1])
        troughs, _ = find_peaks(-data_points[:, 1])
        
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
        df = df.sort_values('Month', key=lambda x: pd.Categorical(x, categories=months, ordered=True))
        
        st.sidebar.header("Customize Colors")
        high_color = st.sidebar.color_picker("High value color", "#00FF00")
        low_color = st.sidebar.color_picker("Low value color", "#FF0000")
        
        def color_cells(val):
            color = high_color if val == "High" else low_color
            return f'background-color: {color}'
        
        styled_df = df.style.applymap(color_cells, subset=['Type'])
        
        st.write("### High and Low Values")
        st.dataframe(styled_df)
        
        insights = generate_insights(df)
        
        st.write("### Graph Interpretation Summary")
        for insight in insights:
            st.write(f"- {insight}")
    else:
        st.write("Could not extract data points from the image. Please try a different image.")
else:
    st.write("Please upload an image to begin analysis.")
