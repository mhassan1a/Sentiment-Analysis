# Sentiment Analysis With LSTM

## Overview
This project explores the relationship between sentiment analysis and emotional dimensions in text data. The hypothesis is that regardless of whether the model predicts the number of stars or the emotional dimensions of the text, the precision of sentiment classification should remain consistent for inputs that share the same sentiment. To test this hypothesis, we train a sentiment analysis model on the Go Emotions dataset, which annotates text with emotional dimensions. We then use this trained model to project Yelp reviews into emotional dimensions. Subsequently, we train an SVM to predict the rating of the Yelp reviews based on emotional embeddings and compare the performance with that of the reference Model trained on Yelp Reviews directly.

## Datasets
- **[Yelp Review](https://huggingface.co/datasets/yelp_review_full)**: The Yelp reviews dataset consists of reviews from Yelp. It is extracted from the Yelp Dataset Challenge 2015 data.
- **[Go Emotions](https://huggingface.co/datasets/go_emotions)**: The GoEmotions dataset contains 58k carefully curated Reddit comments labeled for 27 emotion categories or Neutral. The raw data is included as well as the smaller, simplified version of the dataset with predefined train/val/test splits.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/mhassan1a/Sentiment-Analysis.git
   ```
2. Navigate to the project directory:
   ```
   cd  Sentiment-Analysis

   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. **Data Preparation**: Obtain the Yelp Review and Go Emotions datasets and preprocess them using the provided scripts in the `utils/` directory.
2. **Model Training**: Train the sentiment analysis model using the LSTM network on the Go Emotions dataset. Fine-tune the model parameters to optimize performance.
3. **Projection of Yelp Reviews**: Apply the trained model to project Yelp reviews into emotional dimensions.
4. **SVM Training**: Train an SVM model to predict the rating of Yelp reviews based on emotional embeddings.
5. **Performance Evaluation**: Compare the performance of the SVM model with the reference model trained directly on Yelp reviews.

## Directory Structure
- `src/`
    - `utils/`: Scripts for data preprocessing and auxuliary functions.
    - `models/`: Implementation  models for sentiment analysis.
    - `training.py`
    - `results.ipynb`
- `requirements.txt`: List of Python dependencies for reproducing the environment.
- `README.md`: Overview, installation instructions, and usage guidelines for the project.

## Contributors
- [Mohamed Hassan](https://github.com/mhassan1a)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.