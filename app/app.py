"""
Streamlit application for YouTube video classification using LSTM model.
The app provides interface for:
- Single text classification
- Batch classification from CSV
- Model performance metrics visualization

Author: Pouryare
Date: October 2024
"""

import os
import time
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Video Content Classifier",
    layout="centered",
    initial_sidebar_state="auto",
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stProgress > div > div > div > div {
            background-color: #1f77b4;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts() -> Tuple[Any, Any, Any]:
    """
    Load model artifacts with caching.
    
    Returns:
        Tuple of model, tokenizer, and label encoder
    """
    with st.spinner("Loading model artifacts..."):
        model = load_model(os.path.join(os.getcwd(), 'video_classification_model.keras'))
        tokenizer = joblib.load(os.path.join(os.getcwd(), 'tokenizer.joblib'))
        label_encoder = joblib.load(os.path.join(os.getcwd(), 'label_encoder.joblib'))
    return model, tokenizer, label_encoder

@st.cache_data
def predict_category(
    text: str,
    _model: Any,
    _tokenizer: Any,
    _label_encoder: Any,
    max_len: int = 50
) -> Tuple[str, np.ndarray]:
    """
    Predict category for input text.
    
    Args:
        text: Input text
        _model: Loaded keras model (unhashable)
        _tokenizer: Fitted tokenizer (unhashable)
        _label_encoder: Fitted label encoder (unhashable)
        max_len: Maximum sequence length
        
    Returns:
        Predicted category and prediction probabilities
    """
    # Preprocess text
    sequence = _tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len)
    
    # Get prediction
    pred_probs = _model.predict(padded, verbose=0)
    predicted_class = _label_encoder.inverse_transform([np.argmax(pred_probs[0])])
    
    return predicted_class[0], pred_probs[0]

def plot_prediction_probabilities(probs: np.ndarray, classes: List[str]) -> go.Figure:
    """
    Create bar plot of prediction probabilities.
    
    Args:
        probs: Prediction probabilities
        classes: Class names
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=probs,
            text=np.round(probs * 100, 1),
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Category",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        height=400
    )
    
    return fig

def process_batch_predictions(
    df: pd.DataFrame,
    model: Any,
    tokenizer: Any,
    label_encoder: Any
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Process batch predictions for a DataFrame.
    
    Args:
        df: Input DataFrame with combined_text column
        model: Loaded model
        tokenizer: Fitted tokenizer
        label_encoder: Fitted label encoder
    
    Returns:
        DataFrame with predictions and probability arrays
    """
    predictions = []
    probabilities = []
    
    progress_bar = st.progress(0)
    for i, text in enumerate(df['combined_text']):
        category, probs = predict_category(
            text,
            model,
            tokenizer,
            label_encoder
        )
        predictions.append(category)
        probabilities.append(probs)
        progress_bar.progress((i + 1) / len(df))
    
    df['Predicted_Category'] = predictions
    return df, np.array(probabilities)

def main():
    """Main function to run the Streamlit app."""
    
    st.title("Video Content Classifier")
    st.write("Classify YouTube video content based on title and description")
    
    # Load model artifacts
    model, tokenizer, label_encoder = load_artifacts()
    
    st.divider()
    
    # Single prediction section
    st.subheader("Single Prediction")
    
    with st.form("prediction_form"):
        title = st.text_input("Video Title")
        description = st.text_area("Video Description")
        predict_button = st.form_submit_button("Predict")
        
        if predict_button and (title.strip() or description.strip()):
            # Combine title and description
            full_text = f"{title} {description}".strip()
            
            with st.spinner("Predicting..."):
                # Add progress bar for visual feedback
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Get prediction
                predicted_category, probabilities = predict_category(
                    full_text,
                    model,
                    tokenizer,
                    label_encoder
                )
                
                # Show metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Predicted Category",
                        value=predicted_category,
                    )
                with col2:
                    st.metric(
                        label="Confidence",
                        value=f"{np.max(probabilities)*100:.1f}%",
                    )
                
                # Plot probabilities
                fig = plot_prediction_probabilities(
                    probabilities,
                    label_encoder.classes_
                )
                st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Batch prediction section
    st.subheader("Batch Prediction")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with 'Title' and 'Description' columns",
        type=['csv']
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'Title' not in df.columns or 'Description' not in df.columns:
                st.error("CSV must contain 'Title' and 'Description' columns")
            else:
                with st.spinner("Processing batch predictions..."):
                    # Combine title and description
                    df['combined_text'] = df['Title'] + " " + df['Description']
                    
                    # Process predictions
                    df, probabilities = process_batch_predictions(
                        df,
                        model,
                        tokenizer,
                        label_encoder
                    )
                    
                    # Show results
                    st.write("Prediction Results:")
                    st.dataframe(
                        df[['Title', 'Description', 'Predicted_Category']],
                        hide_index=True
                    )
                    
                    # Download link for results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Show category distribution
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.pie(
                            df,
                            names='Predicted_Category',
                            title='Distribution of Predicted Categories'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Average confidence by category
                        avg_conf = pd.DataFrame({
                            'Category': label_encoder.classes_,
                            'Avg Confidence': np.mean(probabilities, axis=0) * 100
                        })
                        fig = px.bar(
                            avg_conf,
                            x='Category',
                            y='Avg Confidence',
                            title='Average Confidence by Category'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    st.divider()
    
    # Model information section
    st.subheader("Model Information")
    
    with st.expander("Model Details"):
        st.write("Model Architecture Overview:")
        st.write("""
        - Input Layer: Text sequences (max length: 50)
        - Embedding Layer: 100 dimensions
        - Spatial Dropout: 0.2
        - LSTM Layer: 100 units with dropout and recurrent dropout of 0.2
        - Dense Output Layer: Softmax activation with 6 classes
        """)
        
        st.write("Available Categories:")
        categories = pd.DataFrame({
            'Category': label_encoder.classes_,
            'Index': range(len(label_encoder.classes_))
        })
        st.dataframe(categories, hide_index=True)
        
        st.write("Model Input Requirements:")
        st.write("""
        - Maximum sequence length: 50
        - Text input: Combined title and description
        - Text should be in English
        """)

if __name__ == "__main__":
    main()