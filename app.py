from flask import Flask, render_template, request, jsonify
import os
import pickle
from datetime import datetime
from transformers import pipeline
from better_profanity import profanity

app = Flask(__name__)

# Initialize the profanity filter
profanity.load_censor_words()

# Load the tokenizer
try:
    with open('tokenizer1.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    print("✅ Tokenizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading tokenizer: {e}")
    exit()

# Load the BERT fill-mask pipeline
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

# -----------------------------------------------------------
# Function to store user input and prediction in a .txt file
# -----------------------------------------------------------
def store_user_data(input_text, predicted_text, filename='user_data.txt'):
    """
    Append the user input and predicted text to a text file.
    Each record is separated by a line of dashes.
    """
    record = f"{predicted_text}\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(record)


# -----------------------------------------------------------
# Function to combine the original dataset with user data (from .txt file)
# -----------------------------------------------------------
def update_combined_dataset(original_file='metamorphosis_clean.txt', user_file='user_data.txt', output_file='combined_dataset.txt'):
    # Read the original dataset
    if os.path.isfile(original_file):
        with open(original_file, 'r', encoding='utf-8') as f:
            original_data = f.read()
    else:
        original_data = ""
    
    # Read user data from text file
    if os.path.isfile(user_file):
        with open(user_file, 'r', encoding='utf-8') as f:
            user_data = f.read()
    else:
        user_data = ""
    
    # Combine original and user data
    combined_data = original_data + "\n" + user_data
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(combined_data)
    return output_file

# -----------------------------------------------------------
# Function to update the tokenizer using the combined dataset
# -----------------------------------------------------------
def update_tokenizer_from_combined_dataset(combined_file='combined_dataset.txt'):
    try:
        with open(combined_file, 'r', encoding='utf-8') as f:
            combined_text = f.read()
        global tokenizer
        tokenizer.fit_on_texts([combined_text])
        with open('tokenizer1.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)
        print("✅ Tokenizer updated from combined dataset successfully!")
    except Exception as e:
        print(f"❌ Error updating tokenizer: {e}")

# -----------------------------------------------------------
# Function to predict the next word using BERT with profanity filtering
# -----------------------------------------------------------
def predict_next_word_bert(text):
    masked_text = text + " [MASK]."
    predictions = fill_mask(masked_text)
    print(f"\n🔹 Input: {text}")
    for idx, prediction in enumerate(predictions[:3]):
        word = prediction['token_str']
        if not profanity.contains_profanity(word):
            print(f"   ➤ Prediction {idx+1}: {word} ({prediction['score']*100:.2f}%)")
        else:
            print(f"   ➤ Prediction {idx+1}: [Filtered]")
    for prediction in predictions:
        if not profanity.contains_profanity(prediction['token_str']):
            return prediction['token_str']
    return "word"

# -----------------------------------------------------------
# Function to iteratively predict the next num_words words
# -----------------------------------------------------------
def Predict_Next_Words(text, num_words):
    predicted_sentence = text
    for _ in range(num_words):
        next_word = predict_next_word_bert(predicted_sentence)
        predicted_sentence += " " + next_word.strip()
    return predicted_sentence

# -----------------------------------------------------------
# Flask Routes
# -----------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_view():
    text = request.form['text']
    num_words = int(request.form['num_words'])
    try:
        # Generate prediction using the existing function
        predicted_sentence = Predict_Next_Words(text, num_words)
        # Store the user input and prediction in a text file
        store_user_data(text, predicted_sentence)
        # Combine original dataset with new user data
        combined_file = update_combined_dataset()
        # Update the tokenizer based on the combined dataset
        update_tokenizer_from_combined_dataset(combined_file)
        return render_template('index.html', text=text, predicted_sentence=predicted_sentence)
    except Exception as e:
        print(f"❌ Error: {e}")
        return render_template('index.html', text=text, error=f"Error: {str(e)}")

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json()
    text = data['text']
    num_words = int(data['num_words'])
    try:
        predicted_sentence = Predict_Next_Words(text, num_words)
        store_user_data(text, predicted_sentence)
        combined_file = update_combined_dataset()
        update_tokenizer_from_combined_dataset(combined_file)
        return jsonify({'predicted_sentence': predicted_sentence})
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
