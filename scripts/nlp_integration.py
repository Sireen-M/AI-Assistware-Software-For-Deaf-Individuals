from transformers import T5Tokenizer, T5ForConditionalGeneration
import sys

# Load fine-tuned gloss-to-English model (replace path with your model)
tokenizer = T5Tokenizer.from_pretrained("./gloss_to_english_model")
model = T5ForConditionalGeneration.from_pretrained("./gloss_to_english_model")

def gloss_to_sentence(gloss_str):
    input_text = f"translate gloss to English: {gloss_str}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    # You can run like: python nlp_integration.py "I GO STORE YESTERDAY"
    gloss_input = sys.argv[1]
    english_sentence = gloss_to_sentence(gloss_input)
    print(english_sentence)
