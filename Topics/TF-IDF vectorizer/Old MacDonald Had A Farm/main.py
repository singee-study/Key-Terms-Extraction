from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == '__main__':
    vectorizer = TfidfVectorizer(input='filename', use_idf=True, lowercase=True,
                                 analyzer='word', ngram_range=(1, 1),
                                 stop_words=None)
    matrix = vectorizer.fit_transform(["data/dataset/input.txt"])
    print(matrix[(0, 10)])
