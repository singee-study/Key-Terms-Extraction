from lxml import etree
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from collections import Counter

if __name__ == '__main__':
    stdin = open("news.xml", "rb").read()

    lemmatizer = WordNetLemmatizer()
    ignores = set(list(stopwords.words('english')) + list(string.punctuation))

    xml = etree.fromstring(stdin)
    corpus = xml[0]
    for news in corpus:
        head = news[0].text
        text = news[1].text

        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(x) for x in tokens]
        tokens = [x for x in tokens if x and x not in ignores]

        counter = Counter(tokens)

        top5 = counter.most_common(10)
        top5.sort(key=lambda x: x[0], reverse=True)
        top5.sort(key=lambda x: x[1], reverse=True)
        top5 = top5[:5]

        print(head, end=':\n')
        print(" ".join([x[0] for x in top5]))
