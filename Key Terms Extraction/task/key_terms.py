import sys
from lxml import etree
from nltk.tokenize import word_tokenize
from collections import Counter

if __name__ == '__main__':
    stdin = open("news.xml", "rb").read()

    xml = etree.fromstring(stdin)
    corpus = xml[0]
    for news in corpus:
        head = news[0].text
        text = news[1].text

        tokens = word_tokenize(text.lower())
        counter = Counter(tokens)

        top5 = counter.most_common(10)
        top5.sort(key=lambda x: x[0], reverse=True)
        top5.sort(key=lambda x: x[1], reverse=True)
        top5 = top5[:5]

        print(head, end=':\n')
        print(" ".join([x[0] for x in top5]))
