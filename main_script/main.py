import spacy
import re
from collections import Counter
from titlecase import titlecase
import json
import os

from llama_cpp import Llama
from huggingface_hub import hf_hub_download, configure_http_backend
import analysis

nlp = spacy.load("ru_core_news_lg")

project_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(project_dir, "model_llama")
model_path = os.path.join(model_dir, "model-q8_0.gguf")
text_dir = os.path.join(os.path.dirname(project_dir), "test_text")
text_path = os.path.join(text_dir, "text.txt")

print("⏳ Loading LLaMA model...")
llm = Llama(model_path=model_path, verbose=False)
print("✅ Model loaded")

#текст
with open(text_path, 'r', encoding='utf-8') as file:
    text = file.read()
print("✅ Original text loaded\n")

morphology_information = {}
listing_main_words = []
list_all = []
doc_list = []
list_end = []
kolvo = []
rel = {}
ll = []

# Main theme
doc = nlp(text)
for ent in doc.ents:
  doc_list.append(ent.text)
list_all = analysis.morpho(doc, doc_list, list_all)
list_all = analysis.find_lemma(list_all)
doc_list = [titlecase(word) for word in analysis.find_lemma(doc_list)]
list_end.extend(analysis.count_lemma(list_all))
list_end.extend(analysis.count_lemma(doc_list))
list_end = analysis.check_list(analysis.listings([list_end], text)[0])
print("Combined list of candidate topics:", list_end)

# query for main theme
context = analysis.context_model(text, f"Ответ должен быть в формате json под ключом main_theme")
query = f"Выбери главную тему текста из предложенного списка. " \
        f"Ответ выбери из списка. Список: {list_end}. Текст: {context['text']}. " \
        f"Дополнительная информация: {context['additional_info']}"
answer_main_theme = analysis.query_model(llm, query, 1)
print("Main theme response from model: ", answer_main_theme)
answer_main_theme = answer_main_theme["main_theme"]
if len(answer_main_theme[0].split(" "))>=2:
  answer_main_theme = answer_main_theme.split(" ")[:2]
  answer_main_theme = titlecase(answer_main_theme[0])+" "+answer_main_theme[1]
else:
  answer_main_theme = titlecase(answer_main_theme)

# find lists/enumerations in text
listing = analysis.find_connection(text)
if listing is None:
  query = f"Перепиши текст так, чтобы в нем были перечисления. Текст: {text}"
  text = analysis.query_model(llm, query,0)
  print("Rewritten text:\n", text)
  listing = analysis.find_connection(text)

# query for enumerations
str_zapros = ""
for item in listing:
  str_zapros += "Перечисление, для которого нужно подобрать определяемое слово: " + str(item) + ". "
context = analysis.context_model(text, f"Вывод сделай в формате json. Ключей должно быть {len(listing)}. "
                                       f"Значений должно быть {len(listing)}.")
query = f"Определяемое слово - существительное, от которого зависит задаваемое перечисление. " \
        f"Само определяемое слово не является словом из перечисления. {str_zapros}. " \
        f"Текст: {context['text']}. Дополнительная информация: {context['additional_info']}"
answer_few = analysis.query_model(llm, query,len(listing))
print("Enumeration nouns:", answer_few)
answer_few = analysis.json_loads(answer_few)
listing_theme = [listing[i] for i in range(len(answer_few)) if answer_few[i].lower() in answer_main_theme.lower()]
listing_else = [listing[i] for i in range(len(answer_few)) if answer_few[i].lower() not in answer_main_theme.lower()]
listing_else_words = [answer_few[i] for i in range(len(answer_few)) if answer_few[i].lower() not in answer_main_theme.lower()]
listing_theme_description = analysis.listings(listing_theme, text)
listing_else_description = analysis.listings(listing_else, text)
print("Theme listing descriptions:", listing_theme_description)
print("Other listing descriptions:", listing_else_description)

# query for generalized words for main theme listings
if (listing_theme_description):
  str_zapros = ""
  count = 0
  for item in listing_theme_description:
    count +=1
    str_zapros += "Перечисление для которого нужно подобрать обобщающее слово: " + str(item) + ". "
  context = analysis.context_model(text, f"Вывод сделай в формате json. Ключей должно быть {len(listing)}. "
                                         f"Значений должно быть {len(listing)}.")
  query = f"Обощающее слово - слово, которое обощает задаваемое перечисление. {str_zapros}. " \
          f"Текст: {context['text']}. Дополнительная информация: {context['additional_info']}"
  answer_few_main = analysis.query_model(llm, query, count)
  print("Generalized words for main theme listings:", listing_main_words)
  listing_main_words = analysis.json_loads(answer_few_main)

# query for other listings not related to main theme
if (listing_else_words):
  context = analysis.context_model(text, f"Вывод сделай в формате json в виде ответов да/нет. "
                                         f"Ключей должно быть {len(listing_else_words)}. "
                                         f"Значений должно быть {len(listing_else_words)}.")
  query = f"Связаны ли слова {listing_else_words} с термином {answer_main_theme} в тексте? " \
          f"Текст: {context['text']}. Дополнительная информация: {context['additional_info']}"
  answer_2 = analysis.query_model(llm, query, len(listing_else_words))
  print("Filtered other words:", listing_else_words)
  listing_answers = analysis.json_loads(answer_2)
  for i in range(len(listing_answers)):
    if listing_answers[i]=='нет':
      listing_else_words.pop(i)

# normalize all lists
listing_theme_description_end = analysis.normalize_data(listing_theme_description)
listing_else_description_end = analysis.normalize_data(listing_else_description)
listing_main_words_end = analysis.normalize_data(listing_main_words)
listing_else_words_end = analysis.normalize_data(listing_else_words)

# final relation dictionary
if listing_theme_description_end and listing_main_words_end:
  for i in range(len(listing_main_words_end)):
    rel[listing_main_words_end[i]] = listing_theme_description_end[i]
if listing_else_description_end and listing_else_words_end:
  for i in range(len(listing_else_words_end)):
    rel[listing_else_words_end[i]] = listing_else_description_end[i]
data_to_save = {
    "Общая тема текста": answer_main_theme,
    "Главные части текста": rel
}
project_dir_json = os.path.dirname(os.path.abspath(__file__))
project_root_json = os.path.dirname(project_dir_json)
model_dir_json = os.path.join(project_root_json, "test_text")
model_path_json = os.path.join(model_dir_json, "json_result.json")
# Сохраняем в файл
with open(model_path_json, "w", encoding="utf-8") as f:
    json.dump(data_to_save, f, ensure_ascii=False, indent=4)
print("\n✅ Data successfully saved")
