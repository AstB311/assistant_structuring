# AssistantStructuring

## Project Description
**AssistantStructuring** is a tool for analyzing Russian-language texts using NLP and LLM. It automatically identifies the main theme of the text, detects enumerations, generates generalizing words, and creates a structured JSON report.

---

## Installation

1. Clone the repository:  
```bash
git clone https://github.com/yourusername/AssistantStructuring.git
```

2. Navigate to the project folder:
```bash
cd AssistantStructuring
```

3. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Model and text Setup
1. Put the text for analysis in `test_text/text.txt`
2. Place the LLaMA model in the model_llama folder with the name `model-q8_0.gguf`

LLaMA model (8 gb): https://huggingface.co/IlyaGusev/saiga_llama3_8b_gguf/blob/main/model-q8_0.gguf

## Running the Analysis
Run the main script:
```bash
python main.py
```
The analysis result will be saved in `test_text/json_result.json`

## Features
1. Identifies the main theme of the text
2. Detects enumerations
3. Generates generalizing words
4. Produces a structured JSON report
5. Normalizes word forms
