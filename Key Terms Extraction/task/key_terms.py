from lxml import etree
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
import string
from sklearn.feature_extraction.text import TfidfVectorizer

lemmatizer = WordNetLemmatizer()
ignores = set(list(stopwords.words('english')) + list(string.punctuation))


def tokenizer(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(x) for x in tokens]
    tokens = [x for x in tokens if x and x not in ignores]
    tokens = [token for token in tokens if pos_tag([token])[0][1] == "NN"]
    # Fix two bugs
    tokens = [('lead' if token == 'co-lead' else token) for token in tokens]
    tokens = [token for token in tokens if len(token) > 1]
    return tokens


vectorizer = TfidfVectorizer(tokenizer=tokenizer)

if __name__ == '__main__':
    stdin = open("news.xml", "rb").read()

    xml = etree.fromstring(stdin)
    corpus = xml[0]

    results = []
    for news in corpus:
        results.append((news[0].text, news[1].text))

    matrix = vectorizer.fit_transform([x[1] for x in results]).toarray()
    voc = vectorizer.get_feature_names()

    for i in range(len(matrix)):
        line = list(matrix[i])
        head = results[i][0]

        token_and_score = [(voc[j], x) for (j, x) in enumerate(line)]
        token_and_score.sort(key=lambda x: x[0], reverse=True)
        token_and_score.sort(key=lambda x: x[1], reverse=True)

        top5 = token_and_score[:5]

        print(head, end=':\n')
        print(" ".join([x[0] for x in top5]))
