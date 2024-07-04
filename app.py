import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pytesseract
import pandas as pd
from datetime import datetime
import re

def extract_text_and_data(image):
    text = pytesseract.image_to_string(image)
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        data_points = np.array([point[0] for point in largest_contour])
        data_points = data_points[data_points[:, 0].argsort()]
        
        date_pattern = r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b'
        dates = re.findall(date_pattern, text)
        
        return text, data_points, dates
    
    return text, None, []

def map_data_to_dates(data_points, dates):
    if len(dates) < 2:
        return None
    
    start_date = datetime.strptime(dates[0], '%b %Y')
    end_date = datetime.strptime(dates[-1], '%b %Y')
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    x_min, x_max = data_points[:, 0].min(), data_points[:, 0].max()
    x_range = x_max - x_min
    
    mapped_data = []
    for i, date in enumerate(date_range):
        x_pos = x_min + (i / (len(date_range) - 1)) * x_range
        closest_point = data_points[np.argmin(np.abs(data_points[:, 0] - x_pos))]
        mapped_data.append((date, closest_point[1]))
    
    return pd.DataFrame(mapped_data, columns=['Date', 'Value'])

def generate_insights(df):
    insights = []
    
    period_start = df['Date'].iloc[0].strftime('%B %Y')
    period_end = df['Date'].iloc[-1].strftime('%B %Y')
    
    # Overall trend
    first_value = df['Value'].iloc[0]
    last_value = df['Value'].iloc[-1]
    percent_change = ((last_value - first_value) / first_value) * 100
    trend = "increased" if percent_change > 0 else "decreased"
    insights.append(f"From {period_start} to {period_end}, the value {trend} by {abs(percent_change):.2f}%.")

    # Highest and lowest values
    max_row = df.loc[df['Value'].idxmax()]
    min_row = df.loc[df['Value'].idxmin()]
    insights.append(f"The highest value was {max_row['Value']:.2f} in {max_row['Date'].strftime('%B %Y')}, while the lowest was {min_row['Value']:.2f} in {min_row['Date'].strftime('%B %Y')}.")

    # Comparison to average
    avg_value = df['Value'].mean()
    df['Pct_Diff_From_Avg'] = (df['Value'] - avg_value) / avg_value * 100
    above_avg = df[df['Value'] > avg_value]
    below_avg = df[df['Value'] < avg_value]
    
    insights.append(f"The average value from {period_start} to {period_end} was {avg_value:.2f}.")
    
    above_avg_info = [f"{row['Date'].strftime('%B %Y')} ({row['Value']:.2f}, {row['Pct_Diff_From_Avg']:.2f}% above average)" for _, row in above_avg.iterrows()]
    below_avg_info = [f"{row['Date'].strftime('%B %Y')} ({row['Value']:.2f}, {abs(row['Pct_Diff_From_Avg']):.2f}% below average)" for _, row in below_avg.iterrows()]
    
    insights.append(f"{len(above_avg)} months were above average: {', '.join(above_avg_info)}.")
    insights.append(f"{len(below_avg)} months were below average: {', '.join(below_avg_info)}.")

    # Month-to-month changes
    df['Change'] = df['Value'].pct_change() * 100
    max_increase = df.iloc[df['Change'].idxmax()]
    max_decrease = df.iloc[df['Change'].idxmin()]
    insights.append(f"The largest month-to-month increase was {max_increase['Change']:.2f}% from {df.iloc[df['Change'].idxmax() - 1]['Date'].strftime('%B %Y')} to {max_increase['Date'].strftime('%B %Y')}.")
    insights.append(f"The largest month-to-month decrease was {abs(max_decrease['Change']):.2f}% from {df.iloc[df['Change'].idxmin() - 1]['Date'].strftime('%B %Y')} to {max_decrease['Date'].strftime('%B %Y')}.")

    return insights

def style_dataframe(df, high_color, low_color):
    def color_cells(value):
        if value > df['Value'].mean():
            return f'color: {high_color}'
        else:
            return f'color: {low_color}'
    
    return df.style.applymap(color_cells, subset=['Value'])

st.title("Graph Interpreter with OCR")

# Color pickers in sidebar
st.sidebar.header("Customize Colors")
high_color = st.sidebar.color_picker("Pick a color for high values", "#FF0000")
low_color = st.sidebar.color_picker("Pick a color for low values", "#00FF00")

uploaded_file = st.file_uploader("Upload a graph image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Graph', use_column_width=True)
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    text, data_points, dates = extract_text_and_data(image)
    
    if data_points is not None and dates:
        df = map_data_to_dates(data_points, dates)
        
        if df is not None:
            st.write("### Extracted Data")
            styled_df = style_dataframe(df, high_color, low_color)
            st.dataframe(styled_df)
            
            insights = generate_insights(df)
            
            st.write("### Graph Interpretation Summary")
            for insight in insights:
                st.write(f"- {insight}")
        else:
            st.write("Could not map data points to dates. Please check the image quality.")
    else:
        st.write("Could not extract data points or dates from the image. Please try a different image.")
else:
    st.write("Please upload an image to begin analysis.")
