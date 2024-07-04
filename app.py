import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pytesseract
import pandas as pd
from datetime import datetime
import re
from scipy import stats

def preprocess_image(image):
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    return gray

def extract_text_and_structure(image):
    text = pytesseract.image_to_string(image)
    
    title_pattern = r'^.*$'
    date_pattern = r'\b(?:\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(?:\s+\d{4})?)\b'
    value_pattern = r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b'
    
    title = re.search(title_pattern, text, re.MULTILINE).group(0)
    dates = re.findall(date_pattern, text)
    values = [parse_value(v) for v in re.findall(value_pattern, text)]
    
    chart_structure = {
        'title': title,
        'x_axis': min(dates),
        'y_axis': min(values)
    }
    
    return {'title': title, 'dates': dates, 'values': values}, chart_structure

def parse_value(value_str):
    return float(value_str.replace(',', ''))

def classify_time_scale(dates):
    if all(len(date) == 4 for date in dates):
        return 'yearly'
    elif all('Q' in date for date in dates):
        return 'quarterly'
    else:
        return 'monthly'

def extract_data_points(image, chart_structure):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        data_points = np.array([point[0] for point in largest_contour])
        data_points = data_points[data_points[:, 0].argsort()]
        return data_points
    return None

def map_data_to_time(data_points, time_scale, dates):
    if len(dates) < 2:
        return None
    
    start_date = datetime.strptime(dates[0], '%Y' if time_scale == 'yearly' else '%b %Y')
    end_date = datetime.strptime(dates[-1], '%Y' if time_scale == 'yearly' else '%b %Y')
    
    if time_scale == 'yearly':
        date_range = pd.date_range(start=start_date, end=end_date, freq='YS')
    elif time_scale == 'quarterly':
        date_range = pd.date_range(start=start_date, end=end_date, freq='QS')
    else:
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    x_min, x_max = data_points[:, 0].min(), data_points[:, 0].max()
    x_range = x_max - x_min
    
    mapped_data = []
    for i, date in enumerate(date_range):
        x_pos = x_min + (i / (len(date_range) - 1)) * x_range
        closest_point = data_points[np.argmin(np.abs(data_points[:, 0] - x_pos))]
        mapped_data.append((date, closest_point[1]))
    
    return pd.DataFrame(mapped_data, columns=['Date', 'Value'])

def generate_insights(df, time_scale):
    insights = []
    
    period_start = df['Date'].iloc[0].strftime('%Y' if time_scale == 'yearly' else '%B %Y')
    period_end = df['Date'].iloc[-1].strftime('%Y' if time_scale == 'yearly' else '%B %Y')
    
    overall_trend = "increased" if df['Value'].iloc[-1] > df['Value'].iloc[0] else "decreased"
    percent_change = ((df['Value'].iloc[-1] - df['Value'].iloc[0]) / df['Value'].iloc[0]) * 100
    insights.append((f"From {period_start} to {period_end}, the overall trend {overall_trend} by {abs(percent_change):.2f}%.", [overall_trend]))

    max_row = df.loc[df['Value'].idxmax()]
    min_row = df.loc[df['Value'].idxmin()]
    insights.append((f"The highest value was {max_row['Value']:.2f} in {max_row['Date'].strftime('%Y' if time_scale == 'yearly' else '%B %Y')}, while the lowest was {min_row['Value']:.2f} in {min_row['Date'].strftime('%Y' if time_scale == 'yearly' else '%B %Y')}.", ["highest", "lowest"]))

    avg_value = df['Value'].mean()
    df['Pct_Diff_From_Avg'] = (df['Value'] - avg_value) / avg_value * 100
    above_avg = df[df['Value'] > avg_value]
    below_avg = df[df['Value'] < avg_value]
    
    insights.append((f"The average value from {period_start} to {period_end} was {avg_value:.2f}.", []))
    
    above_avg_info = [f"{row['Date'].strftime('%Y' if time_scale == 'yearly' else '%B %Y')} ({row['Value']:.2f}, {row['Pct_Diff_From_Avg']:.2f}% above average)" for _, row in above_avg.iterrows()]
    below_avg_info = [f"{row['Date'].strftime('%Y' if time_scale == 'yearly' else '%B %Y')} ({row['Value']:.2f}, {abs(row['Pct_Diff_From_Avg']):.2f}% below average)" for _, row in below_avg.iterrows()]
    
    insights.append((f"{len(above_avg)} periods were above average: {', '.join(above_avg_info)}.", ["above"]))
    insights.append((f"{len(below_avg)} periods were below average: {', '.join(below_avg_info)}.", ["below"]))

    df['Change'] = df['Value'].pct_change() * 100
    max_increase = df.iloc[df['Change'].idxmax()]
    max_decrease = df.iloc[df['Change'].idxmin()]
    insights.append((f"The largest {time_scale} increase was {max_increase['Change']:.2f}% from {df.iloc[df['Change'].idxmax() - 1]['Date'].strftime('%Y' if time_scale == 'yearly' else '%B %Y')} to {max_increase['Date'].strftime('%Y' if time_scale == 'yearly' else '%B %Y')}.", ["increase"]))
    insights.append((f"The largest {time_scale} decrease was {abs(max_decrease['Change']):.2f}% from {df.iloc[df['Change'].idxmin() - 1]['Date'].strftime('%Y' if time_scale == 'yearly' else '%B %Y')} to {max_decrease['Date'].strftime('%Y' if time_scale == 'yearly' else '%B %Y')}.", ["decrease"]))

    return insights

def color_text(text, words_to_color, color):
    for word in words_to_color:
        text = text.replace(word, f'<span style="color:{color}">{word}</span>')
    return text

def create_insight_box(insight, index):
    insight_text, words_to_color = insight
    colored_text = color_text(insight_text, words_to_color, positive_color if any(word in ["increased", "highest", "above", "increase"] for word in words_to_color) else negative_color)
    
    html_content = f"""
    <div class="insight-box" id="insight-{index}">
        <div class="insight-content">{colored_text}</div>
        <button class="copy-btn" onclick="copyInsight({index})">Copy</button>
    </div>
    """
    return html_content

def display_insights_in_boxes(insights):
    st.write("### Graph Interpretation Summary")
    
    st.markdown("""
    <style>
    .masonry-container {
        column-count: 3;
        column-gap: 1em;
    }
    .insight-box {
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 1em;
        break-inside: avoid;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .insight-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    .insight-content {
        margin-bottom: 10px;
    }
    .copy-btn {
        background-color: #4CAF50;
        border: none;
        color: white;
        padding: 5px 10px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 12px;
        cursor: pointer;
        border-radius: 4px;
        margin-top: 10px;
    }
    @media (max-width: 1200px) {
        .masonry-container {
            column-count: 2;
        }
    }
    @media (max-width: 800px) {
        .masonry-container {
            column-count: 1;
        }
    }
    </style>
    
    <script>
    function copyInsight(index) {
        var insightText = document.querySelector("#insight-" + index + " .insight-content").innerText;
        navigator.clipboard.writeText(insightText).then(function() {
            alert("Insight copied to clipboard!");
        }, function(err) {
            alert("Could not copy text: ", err);
        });
    }
    </script>
    """, unsafe_allow_html=True)

    # Create a masonry layout container
    st.markdown('<div class="masonry-container">', unsafe_allow_html=True)

    for i, insight in enumerate(insights):
        st.markdown(create_insight_box(insight, i), unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


st.title("Trend Chart Reader")

st.sidebar.header("Customize Colors")
positive_color = st.sidebar.color_picker("Pick a color for positive trends", "#FF0000")
negative_color = st.sidebar.color_picker("Pick a color for negative trends", "#00FF00")

show_table = st.sidebar.checkbox("Show data table", False)

uploaded_file = st.file_uploader("Upload a graph image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Graph', use_column_width=True)
    
    processed_image = preprocess_image(image)
    text_elements, chart_structure = extract_text_and_structure(processed_image)
    
    time_scale = classify_time_scale(text_elements['dates'])
    data_points = extract_data_points(processed_image, chart_structure)
    
    if data_points is not None:
        df = map_data_to_time(data_points, time_scale, text_elements['dates'])
        
        if df is not None:
            insights = generate_insights(df, time_scale)
            display_insights_in_boxes(insights)
            
            if show_table:
                st.write("### Extracted Data")
                st.dataframe(df)
        else:
            st.write("Could not map data points to time. Please check the image quality.")
    else:
        st.write("Could not extract data points from the image. Please try a different image.")
else:
    st.write("Please upload an image to begin analysis.")
