import spacy
import re
from collections import Counter
from titlecase import titlecase
import json
from pymorphy3 import MorphAnalyzer
from typing import List, Union, Optional, Dict, Any

nlp = spacy.load("ru_core_news_lg")
morph = MorphAnalyzer()
morphology_information: Dict[str, List[Any]] = {}

def is_noun(word: str) -> bool:
    p = morph.parse(word)[0]
    return 'NOUN' in p.tag

def is_adjective(word: str) -> bool:
    p = morph.parse(word)[0]
    return 'ADJF' in p.tag or 'ADJS' in p.tag

def agree_adjective(adj_lemma: str, noun_word: str) -> str:
    noun = morph.parse(noun_word)[0]
    gender = noun.tag.gender
    number = noun.tag.number
    adj = morph.parse(adj_lemma)[0]
    gram = {}
    if gender:
        gram['gender'] = gender
    if number:
        gram['number'] = number
    agreed = adj.inflect(gram)
    return agreed.word if agreed else adj_lemma

def agree_noun(noun1_word: str, noun2_word: str) -> str:
    noun2 = morph.parse(noun2_word)[0]
    number = noun2.tag.number
    gender = noun2.tag.gender
    lemma2 = noun2.normal_form
    if number == 'plur' or gender == 'neut':
        return lemma2
    if lemma2.endswith(('а', 'я')):
        return lemma2
    return lemma2 + 'а'

def lemma_few(pair: List[str]) -> str:
    if len(pair) != 2:
        return ' '.join([morph.parse(w)[0].normal_form for w in pair])
    word1, word2 = pair
    p1, p2 = morph.parse(word1)[0], morph.parse(word2)[0]
    if is_adjective(word1) and is_noun(word2):
        agreed_adj = agree_adjective(p1.normal_form, word2)
        return f"{agreed_adj} {p2.normal_form}"
    if is_noun(word1) and is_adjective(word2):
        agreed_adj = agree_adjective(p2.normal_form, word1)
        return f"{agreed_adj} {p1.normal_form}"
    if is_noun(word1) and is_noun(word2):
        agreed_noun2 = agree_noun(word1, word2)
        return f"{p1.normal_form} {agreed_noun2}"
    return f"{p1.normal_form} {p2.normal_form}"

def initial_form(listing: List[List[str]]) -> List[List[str]]:
    if not listing:
        return []
    result: List[List[str]] = []
    for item in listing:
        processed: List[str] = []
        for words in item:
            tokens = words.split()
            if len(tokens) > 1:
                processed.append(lemma_few(tokens))
            else:
                token = morph.parse(words)[0]
                processed.append(token.normal_form)
        result.append(processed)
    return result

def morpho(doc: spacy.tokens.Doc, doc_list: List[str], list_all: List[str]) -> List[str]:
    for token in doc:
        if token.is_alpha and token.tag_ not in {"VERB", "PART", "PRON", "ADJ", "ADV"} and token.text not in doc_list:
            key = token.text
            value = token.morph
            if key in morphology_information:
                morphology_information[key].append(value)
            else:
                morphology_information[key] = [value]
    list_all.extend(item for item in morphology_information if morphology_information[item][0])
    return list_all

def find_lemma(list_for_find: Union[str, List[str]]) -> List[str]:
    if isinstance(list_for_find, str):
        list_for_find = [list_for_find]
    lemmas: List[str] = []
    for word in list_for_find:
        doc = nlp(word)
        lemmas.extend([token.lemma_ for token in doc])
    return lemmas

def count_lemma(list_for_count: List[str]) -> List[str]:
    top = Counter(list_for_count).most_common(5)
    return [word for word, _ in top]

def find_connections_and_dependencies(text: str, target_noun: str) -> Optional[str]:
    doc = nlp(text)
    target_noun_lower = target_noun.lower()
    for token in doc:
        if token.pos_ == 'NOUN' and token.text.lower() == target_noun_lower:
            my_dict = {child.dep_: child.text for child in token.children}
            return my_dict.get("nmod") or my_dict.get("appos") or my_dict.get("amod")

def listings(listing: List[List[str]], text: str) -> List[List[str]]:
    listing_description: List[List[str]] = []
    for item in listing:
        pats_new: List[str] = []
        for parts in item:
            target_noun = parts
            connection = find_connections_and_dependencies(text, target_noun)
            formatted_part = titlecase(parts) if parts not in text else parts
            if connection:
                pos_part = text.find(formatted_part)
                pos_conn = text.find(connection)
                if pos_part > pos_conn and pos_conn != -1 and pos_part != -1:
                    pats_new.append(f"{connection} {target_noun}")
                else:
                    pats_new.append(f"{target_noun} {connection}")
            else:
                pats_new.append(target_noun)
        listing_description.append(pats_new)
    return listing_description

def check_list(listing: List[str]) -> List[str]:
    no_duplicates: List[str] = []
    seen: set = set()
    for item in listing:
        item_lower = item.lower()
        if item_lower not in seen:
            seen.add(item_lower)
            no_duplicates.append(item)
    return no_duplicates

def find_connection(text: str) -> Optional[List[List[str]]]:
    doc = nlp(text)
    listing: List[List[str]] = []
    for token in doc:
        new: List[str] = []
        if token.pos_ != "VERB":
            for child in token.children:
                if child.dep_ == "conj":
                    if str(token) not in new:
                        new.append(str(token))
                    new.append(str(child.text))
                    if new not in listing:
                        listing.append(new)
    if listing:
        return listing
    return None

def context_model(text: str, additional_info: Any) -> Dict[str, Any]:
    return {
        "text": text,
        "additional_info": additional_info,
    }

def query_model_error(answer: str, count: int) -> Optional[List[Any]]:
    if len(answer) == 0: return None
    answer = answer[7:-3]
    try:
        result = json.loads(answer)
        if len(result) == count:
            return result
    except json.JSONDecodeError as e:
        print("Error JSON:", e)
        return None

def query_model(llm: Any, prompt: str, count: int) -> Any:
    prompt1 = prompt
    count1 = count
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}]
    )
    if count1 > 0:
        answer_query_model = query_model_error(response["choices"][0]["message"]["content"], count1)
        if answer_query_model is None:
            answer_query_model = query_model(prompt1, count1)
        else:
            return answer_query_model
    else:
        return response["choices"][0]["message"]["content"]

def json_loads(answer: Dict[str, Any]) -> List[Any]:
    answer_f: List[Any] = []
    for item in answer:
        answer_f.append(answer[item])
    return answer_f

def normalize_phrase(data):
    if isinstance(data, list) and any(isinstance(el, list) for el in data):
        return [normalize_phrase(sublist) for sublist in data]
    if isinstance(data, list) and all(isinstance(el, str) for el in data):
        return [normalize_phrase(el) for el in data]
    if isinstance(data, str):
        parts = data.split()
        if len(parts) < 2:
            return morph.parse(parts[0])[0].normal_form
        first_parse = morph.parse(parts[0])[0]
        if 'ADJF' in first_parse.tag:
            adj = parts[0]
            noun = " ".join(parts[1:])
            noun_parsed = morph.parse(noun)[0]
            noun_lemma = noun_parsed.normal_form
            gender = morph.parse(noun_lemma)[0].tag.gender or 'masc'
            adj_parsed = morph.parse(adj)[0]
            adj_inflected = adj_parsed.inflect({gender, 'sing', 'nomn'})
            adj_lemma = adj_inflected.word if adj_inflected else adj_parsed.normal_form
            return f"{adj_lemma} {noun_lemma}"
        else:
            noun1 = parts[0]
            noun2 = " ".join(parts[1:])
            noun1_lemma = morph.parse(noun1)[0].normal_form
            noun2_parsed = morph.parse(noun2)[0]
            noun2_inflected = noun2_parsed.inflect({'sing', 'gent'})
            noun2_word = noun2_inflected.word if noun2_inflected else noun2_parsed.normal_form
            return f"{noun1_lemma} {noun2_word}"
    raise TypeError(f"Unsupported data type: {type(data)}")
def normalize_data(data):
  if not data:
    return data
  if isinstance(data, list) and all(isinstance(el, list) for el in data):
    normalized = []
    for group in data:
      normalized_group = [normalize_phrase(phrase) for phrase in group]
      normalized.append(normalized_group)
    return normalized
  if isinstance(data, list) and all(isinstance(el, str) for el in data):
    return [morph.parse(word)[0].normal_form for word in data]
  raise TypeError(f"Unexpected data type for normalization: {type(data)}")
