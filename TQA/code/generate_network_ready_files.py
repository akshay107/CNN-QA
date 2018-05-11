import os
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize, sent_tokenize
import pickle
import string
from read_json import read_json

class generate_network_ready_files():
    def __init__(self,word2vec_path,processed_data_path,is_test_data,word_vec_size,max_q_length,max_option_length,max_opt_count,max_sent_para,max_words_sent,op_path=None):
        self.processed_data_path = processed_data_path
        self.raw_text_path = os.path.join(processed_data_path,"text_question_sep_files")
        self.is_test_data = is_test_data
        if not os.path.exists(self.raw_text_path):
            read_json_data = read_json(os.path.dirname(processed_data_path), is_test_data)
            read_json_data.read_json_do_sanity_create_closest_sent_()
        if op_path is None:
            op_path = os.path.join(processed_data_path,"one_hot_files")
        if not os.path.exists(op_path):
            os.makedirs(op_path)
        self.word2vec_path = word2vec_path
        self.op_path = op_path
        self.word_vec_size = word_vec_size
        self.num_of_words_in_opt = max_option_length
        self.num_of_words_in_question = max_q_length
        self.num_of_sents_in_closest_para = max_sent_para
        self.num_of_words_in_sent = max_words_sent        
        self.num_of_words_in_closest_sentence = max_sent_para*max_words_sent
        self.num_of_options_for_quest = max_opt_count
        self.lessons_list = self.get_list_of_dirs(self.raw_text_path)
        self.unknown_words_vec_dict = None
        self.unknown_words_vec_dict_file = "unk_word2vec_dict.pkl"
        self.common_files_path = "../common_files"
        if not os.path.exists(self.common_files_path):
            os.makedirs(self.common_files_path)

    def get_list_of_dirs(self,dir_path):
        dirlist = [name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]
        dirlist.sort()
        return dirlist


    def get_list_of_files(self,file_path,file_extension=".txt"):
        filelist = []
        for root, dirs, files in os.walk(file_path):
            for filen in files:
                if filen.endswith(file_extension):
                    filelist.append(filen)
        filelist.sort()
        return filelist

    def handle_unknown_words(self,word):
        fname = self.unknown_words_vec_dict_file
        if self.unknown_words_vec_dict is None:
            print "Dict is none"
            if os.path.isfile(os.path.join(self.common_files_path,fname)):
                print "Dict file exist"
                with open(os.path.join(self.common_files_path,fname), 'rb') as f:
                    self.unknown_words_vec_dict = pickle.load(f)
            else:
                print "Dict file does not exist"
                self.unknown_words_vec_dict = {}
        if self.unknown_words_vec_dict.get(word,None) is not None:
            print "word present in dictionary : ",word
            vec = self.unknown_words_vec_dict.get(word,None)
        else:
            print "word is not present in dictionary : ", word
            vec = np.random.rand(1,self.word_vec_size)
            self.unknown_words_vec_dict[word] = vec
        return vec

    def get_vec_for_word(self,model, word):
        try:
            vec = model[word]
            return vec
        except:
            print "Vector not in model for word",word
            vec = self.handle_unknown_words(word)
            return vec

    def write_vecs_to_file(self,model,raw_data_content,word2vec_file,is_correct_answer_file = False,is_closest_para_file = False):
        all_vec_array = np.array([])
        number_of_words = 0
        break_loop = False
        if is_correct_answer_file:
            word = raw_data_content[0].strip().lower()
            pos = ord(word) -97
            all_vec_array = 0 * np.ones(self.num_of_options_for_quest)
            all_vec_array[pos] = 1

        elif is_closest_para_file:
            all_vec_array = np.zeros((self.num_of_sents_in_closest_para,self.num_of_words_in_sent,self.word_vec_size))
            sents = sent_tokenize(raw_data_content)
            for i in range(len(sents)):
                words = word_tokenize(sents[i])
                words = [w for w in words if w not in string.punctuation]
                # sanity check
                if len(words)>self.num_of_words_in_sent:
                    words = words[:self.num_of_words_in_sent]
                for j in range(len(words)):
                    word = words[j].strip().lower()
                    vec = self.get_vec_for_word(model, word)
                    all_vec_array[i,j,:] = vec

        else:
            for sent in raw_data_content:
                words = word_tokenize(sent)
                words = [w for w in words if w not in string.punctuation]   ## to remove punctuations
                for word in words:
                    word = word.strip().lower()
                    vec = self.get_vec_for_word(model, word)
                    all_vec_array = np.append(all_vec_array, vec)
                    number_of_words+=1
                    if number_of_words>self.num_of_words_in_closest_sentence-1:
                        break_loop = True
                        break
                if break_loop:
                    break

        pickle.dump(all_vec_array, word2vec_file)
        word2vec_file.close()

    def generate_word2vec_for_all(self):

        print 20 * "*"
        print "GENERATING NETWORK READY FILES."

        model = Word2Vec.load_word2vec_format(self.word2vec_path, binary=True)

        for lesson in self.lessons_list:
            l_dir = os.path.join(self.raw_text_path,lesson)
            print ("Lesson : ",lesson)
            op_l_dir = os.path.join(self.op_path,lesson)
            if not os.path.exists(op_l_dir):
                os.makedirs(op_l_dir)
            questions_dir = self.get_list_of_dirs(l_dir)
            questions_dir = [name for name in questions_dir if name.startswith("NDQ")]
            for question_dir in questions_dir:
                file_list = self.get_list_of_files(os.path.join(l_dir,question_dir))
                if not os.path.exists(os.path.join(op_l_dir,question_dir)):
                    os.makedirs(os.path.join(op_l_dir,question_dir))
                print ("Question : ", question_dir)
                for fname in file_list:
                    if fname == "correct_answer.txt":
                        is_correct_answer_file = True
                    else:
                        is_correct_answer_file = False
                    with open(os.path.join(l_dir,question_dir, fname),"r") as f:
                        if fname == 'closest_sent.txt':
                            is_closest_para_file = True
                            try:
                                text = f.readlines()[0]
                                raw_data_content = ""
                                count = 0
                                for s in sent_tokenize(text):
                                    if len(s.split())> self.num_of_words_in_sent:
                                        raw_data_content += " ".join(s.split()[:self.num_of_words_in_sent])
                                        raw_data_content += ". "
                                    else:
                                        raw_data_content += " ".join(s.split())
                                        raw_data_content += " "
                                    count+=1
                                    if count == self.num_of_sents_in_closest_para:
                                        break
                            except:
                                raw_data_content = f.readlines()
                        else:
                            is_closest_para_file = False
                            raw_data_content = f.readlines()

                    f = open(os.path.join(op_l_dir, question_dir, fname[:-4]+".pkl"),"w")
                    self.write_vecs_to_file(model,raw_data_content,f,is_correct_answer_file,is_closest_para_file)
                    f.close()
            print 20*"***"
        print "saving final unknown word2vec dictionary to file"
        f = open(os.path.join(self.common_files_path,self.unknown_words_vec_dict_file), "wb")
        pickle.dump(self.unknown_words_vec_dict, f)
        f.close()
