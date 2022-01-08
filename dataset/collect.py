from requests_html import HTMLSession
session = HTMLSession()
import time
import feedparser
from transformers import MBartTokenizer, MBartForConditionalGeneration

model_name = "IlyaGusev/mbart_ru_sum_gazeta"
tokenizer = MBartTokenizer.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

def summ(article_text):
    input_ids = tokenizer(
        [article_text],
        max_length=600,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"]

    output_ids = model.generate(
        input_ids=input_ids,
        no_repeat_ngram_size=4
    )[0]

    summary = tokenizer.decode(output_ids, skip_special_tokens=True)
    return summary

pages = 11 + 1
file = 'cache'
stop_words = ['Блог', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
             'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']



if __name__ == '__main__':
    rsss = open(file, 'r').readlines()
    iss = 0
    total_hours = 0
    num_lines = sum(1 for line in open(file))
    print('Подсчёт постов...')
    ff = 0
    for rss in rsss:
        d = feedparser.parse(rss)
        for i in d.entries:
            ff = ff + 1
            print(ff)
    print('Готово! ' + str(ff))
    for rss in rsss:
        try:
            d = feedparser.parse(rss)
            for i in d.entries:
                start = time.time()
                iss = iss + 1
                print(str(iss) + ' / ' + str(ff) + ' Осталось примерно ' + str(total_hours) + 'ч')
                c = []

                r = session.get(i.link)
                categoryes = r.html.find('.tm-separated-list__item')
                for category in categoryes:
                    df = False
                    for word in stop_words:
                        if word in category.text:
                            df = True
                    if df is False:
                        c.append(category.text)
                try:
                    article_texts = r.html.find('.article-formatted-body')
                    t = ''
                    for a in article_texts:
                        t = a.text
                    short_article_text = summ(t)
                    with open('train.csv', 'a') as f:
                        cat = ''
                        for s in c:
                            cat = cat + s + ';'
                        f.write(short_article_text.replace(',', '') + ',' + cat + '\n')
                    end = time.time()
                    total_hours = round(((end - start) * (ff - iss)) / 3600)
                except:
                    pass
        except:
            pass
