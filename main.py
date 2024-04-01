import pandas as pd
from gensim.models import Word2Vec

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import brown
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from itertools import islice

nltk.download("brown")
nltk.download("stopwords")
nltk.download("punkt")

from nltk.corpus import stopwords
import string
from nltk.tokenize import RegexpTokenizer
from itertools import product
import os
import pytrec_eval
from sklearn.feature_extraction.text import TfidfVectorizer


def read_golden_standard(filepath):
    # open the file
    with open(filepath) as fh:
        # read the data
        data = fh.read()
        # line by line
        lines = [y for y in data.split("\n")]
        # get the similarity scores by going line by line
        similarity_dic_golden = {"word1": [], "word2": [], "score": []}
        for line in lines[1:]:
            # split by tabs
            try:
                record = line.split("\t")
                word1 = record[0]
                word2 = record[1]
                lex_sim = record[3]
                similarity_dic_golden["word1"].append(word1)
                similarity_dic_golden["word2"].append(word2)
                similarity_dic_golden["score"].append(lex_sim)

                similarity_dic_golden["word1"].append(word2)
                similarity_dic_golden["word2"].append(word1)
                similarity_dic_golden["score"].append(lex_sim)
            except:
                pass

        # create a dataframe
        df = pd.DataFrame(similarity_dic_golden)

        return df


def find_similar(inp_word, data):
    # this function finds similar words given a dataframe and words

    # find the rows with similar words
    r = data[data["word1"] == inp_word]
    # get the similar word
    word_sim = [(word, score) for word, score in r[["word2", "score"]].values.tolist()]
    # sort based on similarity scores
    return sorted(word_sim, key=lambda t: t[1], reverse=True)


# function to increase the size if threshold of 10 is not reached
def increase_size(data, word_list, main_word):

    # already size 10 reached
    if len(word_list) >= 10:
        return word_list

    res_arr = []
    res_arr.extend(word_list[:])
    # loop through the word list to get similar words
    for w, s in word_list:
        similar = find_similar(w, data)
        # extend the final array with similar words
        res_arr.extend(similar)

    # remove if main word is in the similar words
    fin_arr = []
    found_words = set()

    for tup in res_arr:
        word = tup[0]
        if word == main_word:
            continue

        elif word not in found_words:
            fin_arr.append(tup)
            found_words.add(word)

    sorted_list = sorted(fin_arr, key=lambda t: t[1], reverse=True)

    # if greater than 10
    if len(sorted_list) > 10:
        sorted_list = sorted_list[:10]

    return sorted_list


# function to generate the golden truth
def generate_truth(data):
    final_dict = dict()
    unique_words = set(data["word1"].tolist())
    # loop through all the words
    for word in unique_words:
        # similar words for the current word
        sim = find_similar(word, data)
        # if length less than 10 add more values
        if len(sim) < 10:
            sim = increase_size(data, sim, word)

        # if length greater than 10, truncate
        if len(sim) > 10:
            sim = sim[:10]

        # create a dictionary of the golden values
        temp_dict = {}
        for idx, tup in enumerate(sim):
            temp_dict[tup[0]] = 10 - idx

        final_dict[word] = temp_dict
    return final_dict


# function to pre-process the text
def preprocess_corpus(text):
    text = " ".join(text)
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)

    # get stopwords
    stop = set(stopwords.words("english"))
    punctuation = set(string.punctuation)
    additional_punctuation = ['"', '"', "``"]

    punctuation.update(additional_punctuation)

    # tokenizer
    tokens = [tk.lower() for tk in tokens if tk not in stop and tk not in punctuation]

    return tokens


def load_corpus(genre1, genre2):
    # files for the first genre
    first_files = brown.fileids(categories=genre1)
    first_genre_corpus = brown.sents(categories=genre1)
    # files for the second genre
    second_files = brown.fileids(categories=genre2)
    second_genre_corpus = brown.sents(categories=genre2)

    # pre-process corpus
    pre_process_first_corpus = [preprocess_corpus(doc) for doc in first_genre_corpus]
    pre_process_second_corpus = [preprocess_corpus(doc) for doc in second_genre_corpus]

    return pre_process_first_corpus, pre_process_second_corpus


def word2vec(context_window_size, vector_size, corpus, corpus_name):
    name_of_model = f"word2vec.model.{corpus_name}.window_size={context_window_size}.vector_size={vector_size}"
    path_of_model = f"./models/{name_of_model}"

    # Check if the model file exists
    if os.path.exists(path_of_model):
        # Load the existing model
        print(f"Loading existing Word2Vec model: {path_of_model}")
        model = Word2Vec.load(path_of_model)
    else:
        if not os.path.exists("./models"):
            os.makedirs("./models")
        # Train a new Word2Vec model
        print(
            f"Training Word2Vec model for corpus '{corpus_name}', window size: {context_window_size}, vector size: {vector_size}"
        )
        model = Word2Vec(
            sentences=corpus, vector_size=vector_size, window=context_window_size
        )
        model.train(corpus, total_examples=len(corpus), epochs=1000)

        # Save the trained model
        print(f"Saving trained Word2Vec model to: {path_of_model}")
        model.save(path_of_model)

    return model, name_of_model


def get_word2vec_models(corpus, corpus_name):
    models = {}
    context_window_size = [1, 2, 5, 10]
    vector_size = [10, 50, 100, 300]
    for window_size_ind, vector_size_ind in product(context_window_size, vector_size):
        model, name_of_model = word2vec(
            window_size_ind, vector_size_ind, corpus, corpus_name
        )
        models[name_of_model] = model
    return models


def generate_top_k_w2vec(golden, models):

    finalDict = {}

    for name_of_model, model in models.items():
        finalDict[name_of_model] = {}

        for word in golden.keys():
            try:
                similar_words = model.wv.most_similar(word)
                similar_word_scores = {}

                # Iterate over the similar words and their scores
                for idx, tup in enumerate(similar_words[:10]):
                    similar_word = tup[0]
                    score = 10 - idx
                    similar_word_scores[similar_word] = score

                finalDict[name_of_model][word] = similar_word_scores

            except KeyError:
                continue

    return finalDict


def evaluate_retrieval(golden, top10_method):
    avg_ndcg_scores = []
    evaluator = pytrec_eval.RelevanceEvaluator(golden, {"ndcg"})

    for name_of_model, top_10_ind in top10_method.items():

        # Evaluate the top-k rankings using pytrec_eval
        evaluation_output = evaluator.evaluate(top_10_ind)

        # Convert evaluation results to a DataFrame for easier manipulation
        evaluation_df = pd.DataFrame.from_dict(evaluation_output).transpose()

        # Calculate the average nDCG score for the query
        avg_ndcg_score = evaluation_df["ndcg"].mean()

        # Append the query ID and its average nDCG score to the list
        avg_ndcg_scores.append((name_of_model, avg_ndcg_score))

    # Sort the list of average nDCG scores by score in descending order
    sorted_avg_ndcg_scores = sorted(avg_ndcg_scores, key=lambda x: x[1], reverse=True)

    # query ID and average nDCG score of the top-performing query
    final_result = sorted_avg_ndcg_scores[0]

    return final_result


def evaluate_retrieval_tfidf(golden, top10_method):
    avg_ndcg_scores = []
    evaluator = pytrec_eval.RelevanceEvaluator(golden, {"ndcg"})

    evaluation_output = evaluator.evaluate(top10_method)

    evaluation_df = pd.DataFrame.from_dict(evaluation_output).transpose()

    avg_ndcg_score = evaluation_df["ndcg"].mean()

    avg_ndcg_scores.append(avg_ndcg_score)

    final_result = avg_ndcg_scores

    return final_result


def tf_idf_vectorizer(corpus):
    # convert corpus to required form
    input = [" ".join(sentence) for sentence in corpus]
    vectorizer = TfidfVectorizer()
    # the matrix
    matrix = vectorizer.fit_transform(input)
    # features
    feats = vectorizer.get_feature_names_out()
    # data
    data = pd.DataFrame(matrix.toarray(), columns=feats)

    return data


def calc_topk_tf_idf(data, gold_standard):
    top = {}

    # Convert TF-IDF matrix to a dictionary for efficient access
    vects = {col: data.iloc[:2][col].values.reshape(-1, 1) for col in data.iloc[:2].columns}
    corpus_unique = set(data.columns)

    # Loop through gold standard
    for w in  dict(islice(gold_standard.items(),20)).keys():
        top[w] = {}

        # Check if the word is in the TF-IDF matrix
        if w not in corpus_unique:
            continue

        word_vec = vects[w]

        # Calculate cosine similarity with all other words
        similarities = {}
        for word, vector in vects.items():
            if word != w and word in corpus_unique:
                similarity = cosine_similarity(word_vec, vector)[0][0]
                similarities[word] = similarity

        # Sort the similarities and extract top-k
        sorted_similarities = sorted(
            similarities.items(), key=lambda x: x[1], reverse=True
        )
        for idx, (word, similarity) in enumerate(sorted_similarities[:10]):
            top[w][word] = 10 - idx

    return top


def plot(first, second):
    values = [first[1], second[1]]
    labels = [first[0], second[0]]
    plt.figure(figsize=(8, 6))
    plt.ylabel("nDCG Values")
    plt.xlabel("Techniques")
    plt.title("nDCG Values for Different Corpuses")
    plt.xticks(rotation=42)
    plt.bar(labels, values, color=["blue", "orange"])
    plt.show()
    plt.savefig("bargraph.png", format="png", dpi=300)
