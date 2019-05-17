from query_expansion import sentence_retriever_using_w2vec
import os
import pickle
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec


class get_closest_sentences():
    def __init__(self,processed_data_path):
        self.processed_data_path = processed_data_path
        self.raw_text_path = os.path.join(processed_data_path,"text_question_sep_files")
        self.lessons_list = self.get_list_of_dirs(self.raw_text_path)


    def get_list_of_dirs(self,dir_path):
        dirlist = [name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]
        dirlist.sort()
        return dirlist

    def get_list_of_files(self,file_path,file_extension=".txt"):
        filelist = [name for name in os.listdir(file_path) if
                    name.endswith(file_extension) and not os.path.isdir(os.path.join(file_path, name))]
        filelist.sort()
        return filelist

    def convert_list_to_string(self,ip_list):
        op_string = ""
        for sent in ip_list:
            op_string+=" "+sent.strip()
        return op_string


    def get_closest_sentences(self,topic_content, question_content, sent_f_handle):

        lemmatizer = WordNetLemmatizer()
        def sub_routine(text,match_coeff):
            for line in doc_lines:
                line = line.translate(None, string.punctuation)
                line_lemma = " ".join(lemmatizer.lemmatize(w) for w in line.split(" "))
                line_list = [i for i in line_lemma.lower().split() if i not in stop]
                if len(set(l_query) & set(line_list)) > match_coeff:
                    text += " " + line.lower()
            return text

        stop = set(stopwords.words('english'))
        query_string = self.convert_list_to_string(question_content)
        doc_string = self.convert_list_to_string(topic_content)

        query_string = query_string.translate(None, string.punctuation)
        doc_lines = sent_tokenize(doc_string)

        l_query = [lemmatizer.lemmatize(i) for i in query_string.lower().split() if i not in stop]
        text_ = ""
        match_coeff_ = 1
        text = sub_routine(text_,match_coeff_)
        while text_ == text:
            print "Error : less than 1 match"
            match_coeff_ -=1
            text = sub_routine(text_, match_coeff_)
            # break
        sent_f_handle.write(text)
        return text

    def get_query_based_sentences(self, topic_content, question_content, sent_f_handle):
        sent_retr = sentence_retriever_using_w2vec(self.W2V_MODEL)
        closest_sentences = sent_retr.get_related_sentences(topic_content,self.convert_list_to_string(question_content))
        sent_f_handle.write(closest_sentences)

    def generate_closest_sentence(self):

        w2vec_path = "../../word2vec/GoogleNews-vectors-negative300.bin.gz"
        self.W2V_MODEL = Word2Vec.load_word2vec_format(w2vec_path,binary=True)
        self.W2V_MODEL.init_sims(replace=True)

        print 20 * "*"
        print "GENERATING CLOSEST SENTENCE "

        topic_fname = "topics.txt"
        question_fname = "Question.txt"
        f_ext = ".txt"
        sent_closest_to_question_fname = "closest_sent.txt"
        now_run = True
        for lesson in self.lessons_list:
            if lesson == "L_0482":
                now_run = True
            if now_run:
                print "Lesson : ", lesson
                l_dir = os.path.join(self.raw_text_path, lesson)
                with open(os.path.join(l_dir, topic_fname), "r") as f:
                    topic_content = unicode(f.read(), 'iso-8859-1')

                topic_content = topic_content.split("\n")
                topic_content = [t for t in topic_content if t!='']
                questions_dir = self.get_list_of_dirs(l_dir)
                for question_dir in questions_dir:
                    if question_dir.startswith("NDQ"):
                        print "Question : ", question_dir
                        with open(os.path.join(l_dir, question_dir,question_fname), "r") as f:
                            question_content = f.readlines()
                        option = 'a'
                        while os.path.exists(os.path.join(l_dir, question_dir, option + f_ext)):
                            with open(os.path.join(l_dir, question_dir, option + f_ext), "r") as f:
                                opt = f.readlines()
                            question_content.append(self.convert_list_to_string(opt))
                            option = chr(ord(option) + 1)
                        sent_f_handle = open(os.path.join(l_dir,question_dir, sent_closest_to_question_fname), "w")
                        # self.get_closest_sentences(topic_content,question_content,sent_f_handle)
                        self.get_query_based_sentences(topic_content,question_content,sent_f_handle)
                        sent_f_handle.close()


