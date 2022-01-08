from itertools import groupby
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
model_name = "0x7194633/keyt5-base" # or 0x7194633/keyt5-large
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def generate(text, **kwargs):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        hypotheses = model.generate(**inputs, num_beams=5, **kwargs)
    s = tokenizer.decode(hypotheses[0], skip_special_tokens=True)
    s = s.replace('; ', ';').replace(' ;', ';').lower().split(';')
    s = [el for el, _ in groupby(s)]
    return s
    
article = """Reuters сообщил об отмене 3,6 тыс. авиарейсов из-за «омикрона» и погоды
Наибольшее число отмен авиарейсов 2 января пришлось на американские авиакомпании 
SkyWest и Southwest, у каждой — более 400 отмененных рейсов. При этом среди 
отмененных 2 января авиарейсов — более 2,1 тыс. рейсов в США. Также свыше 6400 
рейсов были задержаны."""

print(generate(article, top_p=1.0, max_length=64))  
# ['лабораторная диагностика', 'панкреатит', 'профилактика', 'профилактика заболеваний', 'научно-популярное', 'биотехнологии', 'здоровье']
