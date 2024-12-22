# Multilingual Chatbot

A multilingual translation system that supports translation between Hindi and English using state-of-the-art transformer models.

## Features

- Hindi to English translation
- English to Hindi translation
- User-friendly web interface using Gradio
- Uses Facebook's mBART model for high-quality translations
- Example translations included

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Kartikeya-G121/Multilingual-chatbot.git
cd Multilingual-chatbot
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python translation_app.py
```

## Usage

1. Open the web interface (URL will be displayed when you run the application)
2. Enter text in the input box
3. Select translation direction (Hindi to English or English to Hindi)
4. Click "Submit" to get the translation

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Gradio
- See requirements.txt for complete list 