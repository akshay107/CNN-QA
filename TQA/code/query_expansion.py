import string
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from index_sent import IndexFiles
from retrieve import retrieve_sents
import os
import shutil

class sentence_retriever_using_w2vec():
    def __init__(self,w2v_model=None):
        if w2v_model is None:
            w2vec_path = "/home/cvpr/Debjyoti/docvec/GoogleNews-vectors-negative300.bin.gz"
            self.W2V_MODEL = Word2Vec.load_word2vec_format(w2vec_path,binary=True)
            self.W2V_MODEL.init_sims(replace=True)
        else:
            self.W2V_MODEL = w2v_model
        pass


    def query_expansion_wordnet(self,word_list):
        expanded_query = []
        for word in word_list:
            try:
                for i, j in enumerate(wordnet.synsets(word)):
                    expanded_query += j.lemma_names()
            except:
                try:
                    synsets = wordnet.synsets(word)
                    names = [l.name() for s in synsets for l in s.lemmas()]
                    for name in names:
                        expanded_query += name.split('_')   # phrases are seperated by underscore
                except:
                    break
        expanded_query = map(lambda word: word.encode('utf-8'), expanded_query)
        return expanded_query



    def query_expansion_word2vec(self,word_list):
        expanded_query = []
        for word in word_list:
            expanded_query.append(word)
            try:
                for w in self.W2V_MODEL.most_similar(positive=[word], negative=[], topn=20):
                    if self.W2V_MODEL.similarity(word,w[0])> 0.6:
                        expanded_query.append(w[0])
            except:
                continue
        expanded_query = map(lambda word: word.encode('utf-8'), expanded_query)
        return expanded_query


    def get_related_sentences(self,doc_lines,query):
        # f = open("topics.txt","r")
        # # text = f.read()
        # text = unicode(f.read(), 'iso-8859-1')
        # text = text.replace("\n"," ")
        INDEX_DIR = "IndexFiles.index"
        # doc_lines = sent_tokenize(text)
        base_dir = os.getcwd()
        IndexFiles(doc_lines, base_dir)

        query_string = query.translate(None, string.punctuation)
        lemmatizer = WordNetLemmatizer()
        stop = set(stopwords.words('english'))
        l_query = [lemmatizer.lemmatize(i) for i in query_string.lower().split() if i not in stop]
        # l_query = ' '.join(l_query).lower()

        expanded_query = self.query_expansion_word2vec(l_query)
        expanded_query = ' '.join(expanded_query).lower()
        expanded_query = list(set(expanded_query.replace("_"," ").translate(None, string.punctuation).split()))
        expanded_query = ' '.join(expanded_query).lower()
        # print query,expanded_query
        ret_sent = retrieve_sents(os.path.join(base_dir,INDEX_DIR),expanded_query)
        sent_ind_list = ret_sent.retrieve_sents()
        sent_list = []
        for ind in sent_ind_list:
            sent_list.append(doc_lines[ind])
        shutil.rmtree(os.path.join(base_dir,INDEX_DIR))
        return "\n".join(sent_list)


