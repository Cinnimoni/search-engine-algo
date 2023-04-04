import spacy
import json 

nlp = spacy.load("pl_core_news_sm")

with open("prompts_PL.json", "r") as f:
    json_str = f.read()

# Parse the JSON string into a Python dictionary
data = json.loads(json_str)

categories= ['ślub', 'portret', 'dziecięca', 'ciąża', 'rodzina', 
              'impreza', 'romantyczny', 'zwierzę', 'komercyjny']

ruler = nlp.add_pipe("entity_ruler")
patterns = [{"label": "CATEGORY",  "pattern": [{"TEXT": {"FUZZY": i}}]} for i in categories]

with nlp.select_pipes(enable="tagger"):
    ruler.add_patterns(patterns)

# Create a Doc object from the "text" key of the dictionary
doc = nlp(data["text"])

#entity search
booking = {}
for ent in doc.ents:
    if ent.label_ == 'placeName':
        if 'Location' in booking:
            booking['Location'].append(ent.text)
        else:
            booking['Location'] = [ent.text]
    if ent.label_ == 'CATEGORY':
        if 'Category' in booking:
            booking['Category'].append(ent.text)
        else:
            booking['Category'] = [ent.text]

for token in doc:
    if token.pos_ == "NUM":
        if 'Price' in booking:
            booking['Price'].append(token.text)
        else:
            booking['Price'] = [token.text]

if not booking:
    raise FileNotFoundError("""Nie znaleziono żadnych filtrów. Sprawdź pisownie wyrazów lub spróbuj sprecyzować swoje wyszukiwanie.""")
else:
    print(booking)
