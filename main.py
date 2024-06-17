import pandas as pd
from transformers import pipeline

# Load the comments from a CSV file
hotel_reviews = pd.read_csv('database/only_en_vi.csv')
# print('Dataframe inputted!!!')

# Define the classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define the topics for classification
topics_vi = ['dịch vụ', 'vị trí', 'sạch sẽ', 'phòng', 'giá trị']
topics_en = ['service', 'location', 'cleanliness', 'room', 'value']

# Create a mapping from Vietnamese to English labels
vi_to_en = {
    'dịch vụ': 'service',
    'vị trí': 'location',
    'sạch sẽ': 'cleanliness',
    'phòng': 'room',
    'giá trị': 'value'
}

# Function to classify text and return label and confidence score
def classify_text(text, topics):
    result = classifier(text, topics)
    label = result['labels'][0]
    score = result['scores'][0]
    return label, score

# Process each comment in the dataframe
labels = []
scores = []

for _, row in hotel_reviews.iterrows():
    try:
        comment = str(row['comment'])  # Convert to string
        language = row['language']
        
        if language == 'vi':
            label, score = classify_text(comment, topics_vi)
            label = vi_to_en[label]  # Convert label to English
        elif language == 'en':
            label, score = classify_text(comment, topics_en)
        else:
            label, score = 'unknown', None
    except Exception as e:
        print(f"Error processing row {row.name}: {e}")
        label, score = 'unknown', None

    labels.append(label)
    scores.append(score)

# Add new columns to the dataframe
hotel_reviews['label'] = labels
hotel_reviews['confidence_score'] = scores

# Save the updated dataframe to a new CSV file
hotel_reviews.to_csv('database/hotel_reviews_with_labels.csv', index=False)
print('Labels and confidence scores added to dataframe and saved to CSV!!!')
