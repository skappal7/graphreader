import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pytesseract
import pandas as pd
from datetime import datetime
import re
from scipy import stats

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

def calculate_trend(values):
    x = np.arange(len(values))
    slope, _, _, _, _ = stats.linregress(x, values)
    return "increasing" if slope > 0 else "decreasing"

def detect_seasonality(df):
    if len(df) >= 24:  # At least 2 years of data
        yearly_diff = df['Value'] - df['Value'].shift(12)
        return yearly_diff.autocorr() > 0.5
    return False

def identify_outliers(df):
    z_scores = np.abs(stats.zscore(df['Value']))
    return df[z_scores > 3]

def generate_insights(df):
    insights = []
    
    period_start = df['Date'].iloc[0].strftime('%B %Y')
    period_end = df['Date'].iloc[-1].strftime('%B %Y')
    
    # Overall trend
    overall_trend = calculate_trend(df['Value'])
    percent_change = ((df['Value'].iloc[-1] - df['Value'].iloc[0]) / df['Value'].iloc[0]) * 100
    insights.append((f"From {period_start} to {period_end}, the overall trend was {overall_trend} with a {abs(percent_change):.2f}% change.", [overall_trend]))

    # Quarterly trends
    df['Quarter'] = df['Date'].dt.to_period('Q')
    quarterly_trends = df.groupby('Quarter')['Value'].apply(lambda x: calculate_trend(x))
    insights.append((f"Quarterly trends: " + ', '.join([f"{q}: {t}" for q, t in quarterly_trends.items()]), []))

    # Highest and lowest values
    max_row = df.loc[df['Value'].idxmax()]
    min_row = df.loc[df['Value'].idxmin()]
    insights.append((f"The highest value was {max_row['Value']:.2f} in {max_row['Date'].strftime('%B %Y')}, {(max_row['Value']/df['Value'].mean() - 1)*100:.2f}% above the mean.", ["highest"]))
    insights.append((f"The lowest value was {min_row['Value']:.2f} in {min_row['Date'].strftime('%B %Y')}, {(1 - min_row['Value']/df['Value'].mean())*100:.2f}% below the mean.", ["lowest"]))

    # Comparison to average
    avg_value = df['Value'].mean()
    df['Pct_Diff_From_Avg'] = (df['Value'] - avg_value) / avg_value * 100
    above_avg = df[df['Value'] > avg_value]
    below_avg = df[df['Value'] < avg_value]
    
    insights.append((f"The average value from {period_start} to {period_end} was {avg_value:.2f}.", []))
    
    above_avg_info = [f"{row['Date'].strftime('%B %Y')} ({row['Value']:.2f}, {row['Pct_Diff_From_Avg']:.2f}% above average)" for _, row in above_avg.iterrows()]
    below_avg_info = [f"{row['Date'].strftime('%B %Y')} ({row['Value']:.2f}, {abs(row['Pct_Diff_From_Avg']):.2f}% below average)" for _, row in below_avg.iterrows()]
    
    insights.append((f"{len(above_avg)} months were above average: {', '.join(above_avg_info)}.", ["above"]))
    insights.append((f"{len(below_avg)} months were below average: {', '.join(below_avg_info)}.", ["below"]))

    # Month-to-month changes
    df['Change'] = df['Value'].pct_change() * 100
    max_increase = df.iloc[df['Change'].idxmax()]
    max_decrease = df.iloc[df['Change'].idxmin()]
    insights.append((f"The largest month-to-month increase was {max_increase['Change']:.2f}% from {df.iloc[df['Change'].idxmax() - 1]['Date'].strftime('%B %Y')} to {max_increase['Date'].strftime('%B %Y')}.", ["increase"]))
    insights.append((f"The largest month-to-month decrease was {abs(max_decrease['Change']):.2f}% from {df.iloc[df['Change'].idxmin() - 1]['Date'].strftime('%B %Y')} to {max_decrease['Date'].strftime('%B %Y')}.", ["decrease"]))

    # Seasonality
    if detect_seasonality(df):
        insights.append(("Seasonal patterns detected in the data.", ["seasonal"]))

    # Moving averages
    df['MA3'] = df['Value'].rolling(window=3).mean()
    df['MA6'] = df['Value'].rolling(window=6).mean()
    last_ma3 = df['MA3'].iloc[-1]
    last_ma6 = df['MA6'].iloc[-1]
    insights.append((f"The 3-month moving average ended at {last_ma3:.2f}, while the 6-month moving average ended at {last_ma6:.2f}.", []))

    # Outliers
    outliers = identify_outliers(df)
    if not outliers.empty:
        outlier_info = [f"{row['Date'].strftime('%B %Y')} ({row['Value']:.2f})" for _, row in outliers.iterrows()]
        insights.append((f"Outliers detected in {len(outliers)} months: {', '.join(outlier_info)}", ["outliers"]))

    # Correlation with time
    time_correlation = df['Value'].corr(pd.to_numeric(df.index))
    insights.append((f"The correlation between values and time is {time_correlation:.2f}, indicating a {abs(time_correlation):.2f} {'strong' if abs(time_correlation) > 0.7 else 'moderate' if abs(time_correlation) > 0.5 else 'weak'} {overall_trend} trend.", ["correlation"]))

    return insights

def color_text(text, words_to_color, color):
    for word in words_to_color:
        text = text.replace(word, f'<span style="color:{color}">{word}</span>')
    return text

st.title("Enhanced Graph Interpreter with OCR")

# Color pickers in sidebar
st.sidebar.header("Customize Colors")
positive_color = st.sidebar.color_picker("Pick a color for positive trends", "#FF0000")
negative_color = st.sidebar.color_picker("Pick a color for negative trends", "#00FF00")

# Optional table display
show_table = st.sidebar.checkbox("Show data table", False)

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
            insights = generate_insights(df)
            
            st.write("### Graph Interpretation Summary")
            for insight, words_to_color in insights:
                colored_text = color_text(insight, words_to_color, positive_color if any(word in ["increasing", "highest", "above", "increase"] for word in words_to_color) else negative_color)
                st.markdown(colored_text, unsafe_allow_html=True)
            
            if show_table:
                st.write("### Extracted Data")
                st.dataframe(df)
        else:
            st.write("Could not map data points to dates. Please check the image quality.")
    else:
        st.write("Could not extract data points or dates from the image. Please try a different image.")
else:
    st.write("Please upload an image to begin analysis.")
