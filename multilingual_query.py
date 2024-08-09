from googletrans import Translator
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the Translator
translator = Translator()

def translate_to_english(text):
    try:
        # Detect the language and translate to English
        translated = translator.translate(text, dest='en')
        return translated.text
    except Exception as e:
        print(f"Translation Error: {e}")
        return text  # Return the original text if translation fails