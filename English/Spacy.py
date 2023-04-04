import spacy
import json 

nlp = spacy.load("en_core_web_sm")

categories = ['wedding', 'portrait', 'child', 'pregnancy', 
              'family', 'event', 'romantic', 'animal', 'business']

ruler = nlp.add_pipe("entity_ruler")
patterns = [{"label": "CATEGORY",  "pattern": [{"TEXT": {"FUZZY": i}}]} for i in categories]

with nlp.select_pipes(enable="tagger"):
    ruler.add_patterns(patterns)

with open("prompts.json", "r") as f:
    json_str = f.read()

# Parse the JSON string into a Python dictionary
data = json.loads(json_str)

# Create a Doc object from the "text" key of the dictionary
doc = nlp(data["text"])

#entity search
booking = {}
booking = {}
for ent in doc.ents:
    if ent.label_ == 'GPE':
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
    raise FileNotFoundError("""No tags were found. Make sure the words aren't misspelled 
    or try more specific description.""")
else:
    print(booking)
