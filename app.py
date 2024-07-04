import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pytesseract
import pandas as pd
from datetime import datetime
import re
from scipy import stats

# [Previous functions remain the same]

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
    
    # CSS for 3D boxes and copy button
    st.markdown("""
    <style>
    .insight-box {
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .insight-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
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
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
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

    for i, insight in enumerate(insights):
        st.markdown(create_insight_box(insight, i), unsafe_allow_html=True)

st.title("Enhanced Graph Interpreter with OCR")

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
