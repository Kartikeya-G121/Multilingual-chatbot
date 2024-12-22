import gradio as gr
from transformers import MBartForConditionalGeneration, MBartTokenizer
import torch
from multiprocessing import freeze_support

# Initialize translation models
print("Loading translation models...")

def load_model():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBartTokenizer.from_pretrained(model_name)
    return model, tokenizer

def translate_text(text, direction):
    try:
        model, tokenizer = load_model()
        
        if direction == "Hindi to English":
            tokenizer.src_lang = "hi_IN"
            encoded = tokenizer(text, return_tensors="pt", padding=True)
            generated_tokens = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
                max_length=128
            )
            return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        else:  # English to Hindi
            tokenizer.src_lang = "en_XX"
            encoded = tokenizer(text, return_tensors="pt", padding=True)
            generated_tokens = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"],
                max_length=128
            )
            return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    except Exception as e:
        return f"Error in translation: {str(e)}"

# Create Gradio interface
iface = gr.Interface(
    fn=translate_text,
    inputs=[
        gr.Textbox(label="Enter text to translate", lines=5),
        gr.Radio(["Hindi to English", "English to Hindi"], label="Translation Direction", value="Hindi to English")
    ],
    outputs=gr.Textbox(label="Translation", lines=5),
    title="Hindi-English Translation",
    description="Translate text between Hindi and English. Type or paste your text and select the translation direction.",
    examples=[
        ["मैं एक छात्र हूं", "Hindi to English"],
        ["I am a student", "English to Hindi"],
        ["नमस्ते दुनिया", "Hindi to English"],
        ["Hello World", "English to Hindi"]
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    print("Starting translation interface...")
    freeze_support()  # Add support for multiprocessing
    iface.launch(share=True) 