# text to keywords
Trained T5-base and T5-large model for creating keywords from text.
Supported languages: ru


[Pretraining Large version](https://huggingface.co/0x7194633/keyt5-large)
|
[Pretraining Base version](https://huggingface.co/0x7194633/keyt5-base)

[habr article](https://habr.com/ru/post/599715/)

# Usage
Example usage (the code returns a list with keywords. duplicates are possible):

[![Try Model Training In Colab!](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/0x7o/notebooks/blob/main/T5/keyT5_use.ipynb)

```
pip install transformers sentencepiece
```

```python
from itertools import groupby
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
model_name = "0x7194633/keyt5-large" # or 0x7194633/keyt5-base
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def generate(text, **kwargs):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        hypotheses = model.generate(**inputs, num_beams=5, **kwargs)
    s = tokenizer.decode(hypotheses[0], skip_special_tokens=True)
    s = s.replace('; ', ';').replace(' ;', ';').lower().split(';')[:-1]
    s = [el for el, _ in groupby(s)]
    return s

article = """Reuters сообщил об отмене 3,6 тыс. авиарейсов из-за «омикрона» и погоды
Наибольшее число отмен авиарейсов 2 января пришлось на американские авиакомпании 
SkyWest и Southwest, у каждой — более 400 отмененных рейсов. При этом среди 
отмененных 2 января авиарейсов — более 2,1 тыс. рейсов в США. Также свыше 6400 
рейсов были задержаны."""

print(generate(article, top_p=1.0, max_length=64))  
# ['авиаперевозки', 'отмена авиарейсов', 'отмена рейсов', 'отмена авиарейсов', 'отмена рейсов', 'отмена авиарейсов']
```
# Training
To teach the keyT5-base and keyT5-large models, you will need a table in csv format, like this:

KeyT5 models were trained on ~7000 compressed habr.com articles. [data.csv](https://github.com/0x7o/text2keywords/blob/main/dataset/train.csv) [collect.py](https://github.com/0x7o/text2keywords/blob/main/dataset/collect.py)
Exclusively supports the Russian language!
| X | Y |
|:--:|:--:|
| Some text that is fed to the input | The text that should come out |
| Some text that is fed to the input | The text that should come out |

Go to the training notebook and learn more about it:

[![Try Model Training In Colab!](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/0x7o/notebooks/blob/main/T5/keyT5_train.ipynb)

