import json
import os
import shutil
import nltk
from nltk.tokenize import word_tokenize

class read_json():
    def __init__(self,json_dir,op_dir = None):
        self.json_dir = json_dir
        if not op_dir:
            op_dir = os.path.join(json_dir,"processed_data","text_question_sep_files")
        if not os.path.exists(op_dir):
            os.makedirs(op_dir)
        self.op_dir = op_dir
        self.json_file_list = self.get_list_of_files(self.json_dir,file_extension=".json")
        self.option_list = ['distractor1','distractor2','distractor3','correct_answer']

    def get_list_of_dirs(self,dir_path):
        dirlist = [name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]
        dirlist.sort()
        return dirlist

    def get_list_of_files(self,dir_path,file_extension=".json"):
        filelist = [name for name in os.listdir(dir_path) if name.endswith(file_extension) and  not os.path.isdir(os.path.join(dir_path, name))]
        filelist.sort()
        return filelist

    def read_content(self):

        print 20*"*"
        print "READING JSON CONTENT"

        for f in self.json_file_list:
            with open(os.path.join(self.json_dir,f), 'r') as f:
                data = json.load(f)
            batches = [data[x:x+50] for x in range(0, len(data), 50)]
            for i in range(len(batches)):
                print "Batch : ",i
                l_dir = os.path.join(self.op_dir,str(i+1))
                if not os.path.exists(l_dir):
                    os.makedirs(l_dir)
                for j in range(len(batches[i])):
                    q_dir = os.path.join(l_dir,str(j+1))
                    os.makedirs(q_dir)
                    question = batches[i][j]['question']
                    question = question.encode('ascii', 'ignore').decode('ascii')
                    question_f = "Question"+ ".txt"
                    question_f_handle = open(os.path.join(q_dir, question_f), 'w')
                    question_f_handle.write(question+"\n")
                    question_f_handle.close()

                    support = batches[i][j]['support']
                    support = support.encode('ascii', 'ignore').decode('ascii')
                    support_f = "support"+ ".txt"
                    support_f_handle = open(os.path.join(q_dir, support_f), 'w')
                    support_f_handle.write(support+"\n")
                    support_f_handle.close()

                    option_list = []

                    option = 'a'
                    for opt in self.option_list:
                        option_f = option+".txt"
                        option_text = batches[i][j][opt]
                        option_text = option_text.encode('ascii', 'ignore').decode('ascii')
                        option_f_handle = open(os.path.join(q_dir, option_f), 'w')
                        option_f_handle.write(option_text)
                        option_f_handle.close()
                        option = chr(ord(option) + 1)


                    correct_answer = batches[i][j]['correct_answer']
                    correct_answer = correct_answer.encode('ascii', 'ignore').decode('ascii')
                    corr_f_handle = open(os.path.join(q_dir, "correct_answer.txt"), 'w')
                    corr_f_handle.write(correct_answer + "\n")
                    corr_f_handle.close()
