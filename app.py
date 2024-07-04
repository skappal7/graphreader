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
    # [This function remains unchanged]
    pass

def map_data_to_dates(data_points, dates):
    # [This function remains unchanged]
    pass

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
    # [This function remains unchanged]
    pass

# [The rest of the Streamlit app code remains largely unchanged]
