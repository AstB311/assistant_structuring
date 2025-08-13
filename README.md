# AssistantStructuring

Project Description: AssistantStructuring is a tool for analyzing Russian-language texts using NLP and LLM. It automatically identifies the main theme of the text, detects enumerations, generates generalizing words, and creates a structured JSON report.

Installation: Clone the repository with `git clone https://github.com/yourusername/AssistantStructuring.git`. Navigate to the project folder with `cd AssistantStructuring`. Create a virtual environment using `python -m venv venv` and activate it. On Linux/Mac: `source venv/bin/activate`, on Windows: `venv\Scripts\activate`. Install dependencies with `pip install -r requirements.txt`.

Model Setup: Download the SpaCy Russian language model using `python -m spacy download ru_core_news_lg`. Place the LLaMA model in the `model_llama` folder with the name `model-q8_0.gguf`. Put the text for analysis in the file `test_text/text.txt`.

Running the Analysis: Run the script with `python main.py`. The analysis result will be saved in `test_text/json_result.json`.

Project Structure: `model_llama` folder contains the LLaMA model, `test_text` contains input texts and JSON results, `main.py` is the main analysis script, `analysis.py` contains auxiliary text analysis functions, and `requirements.txt` lists project dependencies.

Features: Identifies the main theme of the text, detects enumerations, generates generalizing words, produces a structured JSON report, and normalizes word forms.

Example JSON Output: {"Main theme of the text": "Main Topic", "Main parts of the text": {"Generalizing Word 1": ["Item 1", "Item 2"], "Generalizing Word 2": ["Item 3", "Item 4"]}}.
