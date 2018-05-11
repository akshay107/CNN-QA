import os
import numpy as np
import json
from read_json import read_json
import glob
import re
from keras.models import Model

class generate_result():

    def __init__(self,ndq_test_data):
        self.ndq_test_data = ndq_test_data
        pass
    
    def get_forbidden_questions_tf(self):
        # Returns list of questions which are not true-false
        l = []
        path = self.ndq_test_data.processed_data_path.replace("one_hot_files","text_question_sep_files")
        for lesson in self.ndq_test_data.lessons_list:
            l_dir = os.path.join(path,lesson)
            questions_dir = self.ndq_test_data.get_list_of_dirs(l_dir)
            questions_dir = [name for name in questions_dir if name.startswith("NDQ")]
            for question_dir in questions_dir:
                g = open(os.path.join(path,lesson,question_dir,"Question.txt"),"r").read()
                try:
                    ans = open(os.path.join(path,lesson,question_dir,"a.txt"),"r").read()
                except:
                    continue
                doc = open(os.path.join(path,lesson,question_dir,"closest_sent_try.txt"),"r").read()
                if ans!="true":
                    l.append(question_dir)
        return l
    
    def get_forbidden_questions_mcq(self):
        # Returns list of questions which are either forbidden or true-false
        l = []
        path = self.ndq_test_data.processed_data_path.replace("one_hot_files","text_question_sep_files")
        for lesson in self.ndq_test_data.lessons_list:
            l_dir = os.path.join(path,lesson)
            questions_dir = self.ndq_test_data.get_list_of_dirs(l_dir)
            questions_dir = [name for name in questions_dir if name.startswith("NDQ")]
            for question_dir in questions_dir:
                try:
                    string = open(sorted(glob.glob(os.path.join(path,lesson,question_dir)+"/[a-z].txt"))[-1],"r").read()
                except:
                    l.append(question_dir)
                if 'the above' in string:
                    l.append(question_dir)
                elif len(re.findall(r"^both [a-z] (and|&) [a-z]",string))>0:
                    l.append(question_dir)
                elif len(re.findall(r"^both \([a-z]\) (and|&) \([a-z]\)",string))>0:
                    l.append(question_dir)
                try:
                    ans = open(os.path.join(path,lesson,question_dir,"a.txt"),"r").read()
                except:
                    continue
                if ans=="true":
                    l.append(question_dir)
        return l
        
    def predict_options_one_by_one(self,ndq_model,is_test_data = False):
        # Two dictionaries are prepared for true-false and mcq questions. The value of threshold 
        # can be varied on training set to get the optimum threshold.  
        ndq_acc_pred = 0
        ndq_total_pred = 0    
        total_missed_ques = 0
        options_list = ["a", "b", "c", "d", "e", "f", "g"]
        quest_ans_dict = {}
        quest_mcq_dict = {}
        quest_tf_dict = {}
        l_part2 = []
        intermediate_model  = Model(inputs=ndq_model.input, outputs = ndq_model.layers[-2].output)
        forbidden_list_mcq = self.get_forbidden_questions_mcq()
        forbidden_list_tf = self.get_forbidden_questions_tf()
        print(len(forbidden_list_mcq))
        
        for lesson in self.ndq_test_data.lessons_list:
            l_dir = os.path.join(self.ndq_test_data.processed_data_path, lesson)
            #print ("Lesson : ",lesson)
            questions_dir = self.ndq_test_data.get_list_of_dirs(l_dir)
            questions_dir_tf = [name for name in questions_dir if name.startswith("NDQ") and name not in forbidden_list_tf]
            questions_dir_allow = [name for name in questions_dir if name.startswith("NDQ") and name not in forbidden_list_mcq]
            questions_dir_forbid = [name for name in questions_dir if name.startswith("NDQ") and name in forbidden_list_mcq and name in forbidden_list_tf]
            
            for question_dir in questions_dir_allow:
                flag = False
                question_dir_path = os.path.join(l_dir, question_dir)
                #print ("Question : ", question_dir)
                try :
                    options_mat,max_options = self.ndq_test_data.read_options_files(question_dir_path)
                    question_mat = self.ndq_test_data.read_question_file(question_dir_path)
                    sent_mat = self.ndq_test_data.read_sentence_file(question_dir_path)
                except:
                    total_missed_ques+=1
                    flag = True
                    print ("Lesson : ", lesson)
                    print ("Question : ", question_dir)
                    print "ERROR! THIS QUESTION HAS ERROR"
                
                # print "max option",max_options
                if max_options == 1 :
                    print ("max option is 1")
                    max_options = self.ndq_test_data.num_of_options_for_quest - 1
                pred_options_arr = ndq_model.predict([question_mat, sent_mat, options_mat])

                #Get maximum only from specified options not from complete list
                if flag==False:
                    pred_opt = np.argmax(pred_options_arr[0,:max_options])
                else:
                    pred_opt = 0

                if not is_test_data:
                    correct_ans_mat = self.ndq_test_data.read_correct_ans_file(question_dir_path,max_options)
                    correct_ans_mat[np.where(correct_ans_mat == -1)] = 0
                    corr_opt = np.where(correct_ans_mat == 1)[1][0]
                    if corr_opt == pred_opt:
                        ndq_acc_pred+=1
                ndq_total_pred+=1
                quest_ans_dict[question_dir] = options_list[pred_opt]
                quest_mcq_dict[question_dir] = options_list[pred_opt]
                quest_tf_dict[question_dir] = "h"
            
            for question_dir in questions_dir_forbid:
                flag = False
                thresh = 0.3 #threshold for forbidden questions.. try different values from 0 to 1.
                question_dir_path = os.path.join(l_dir, question_dir)
                #print ("Question : ", question_dir)
                try :
                    options_mat,max_options = self.ndq_test_data.read_options_files(question_dir_path)
                    question_mat = self.ndq_test_data.read_question_file(question_dir_path)
                    sent_mat = self.ndq_test_data.read_sentence_file(question_dir_path)
                except:
                    total_missed_ques+=1
                    flag = True
                    print ("Lesson : ", lesson)
                    print ("Question : ", question_dir)
                    print "ERROR! THIS QUESTION HAS ERROR"
                
                # print "max option",max_options
                if max_options == 1 :
                    print ("max option is 1 in forbid")
                    max_options = self.ndq_test_data.num_of_options_for_quest - 1
                #pred_options_arr = ndq_model.predict([question_mat, sent_mat, options_mat])
                                   
                pred_options_arr = intermediate_model.predict([question_mat, sent_mat, options_mat])
                #Get maximum only from specified options not from complete list
                pred_options_arr = pred_options_arr[:,:max_options]
                last_option_path = question_dir_path.replace("one_hot_files","text_question_sep_files")
                #print pred_options_arr.shape
                #print last_option_path
                if flag==False:
                    string = open(sorted(glob.glob(last_option_path+"/[a-z].txt"))[-1],"r").read()
                    #print string
                    string = string.lower()
                    if 'all of the above' in string or 'none of the above' in string:
                        if np.abs(np.max(pred_options_arr)-np.min(pred_options_arr))<thresh:
                            pred_opt = max_options
                        else:
                            pred_opt = np.argmax(pred_options_arr)
                    elif 'two of the above' in string:
                        if np.abs(np.sort(pred_options_arr)[0,-1]-np.sort(pred_options_arr)[0,-2])<thresh:
                            #print "condition 2 entering if"
                            pred_opt = max_options
                        else:
                            pred_opt = np.argmax(pred_options_arr)
                    elif len(re.findall(r"^both [a-z] (and|&) [a-z]",string))>0 or len(re.findall(r"^both \([a-z]\) (and|&) \([a-z]\)",string))>0:
                        try:
                            opt1 = re.findall(r"^both ([a-z]) (and|&) ([a-z])",string)[0][0]
                            opt2 = re.findall(r"^both ([a-z]) (and|&) ([a-z])",string)[0][2]
                        except:
                            opt1 = re.findall(r"^both \(([a-z])\) (and|&) \(([a-z])\)",string)[0][0]
                            opt2 = re.findall(r"^both \(([a-z])\) (and|&) \(([a-z])\)",string)[0][2]
                        ind1 = options_list.index(opt1)
                        ind2 = options_list.index(opt2)
                        if np.abs(pred_options_arr[0,ind1]-pred_options_arr[0,ind2])<thresh:
                            #print "condition 3 entering if"
                            pred_opt = max_options
                        else:
                            pred_opt = np.argmax(pred_options_arr)
                    elif 'any of the above' in string:
                        pred_opt = max_options
                    else:
                        print string
                        print question_dir
                        pred_opt = 0 # to deal with spelling mistakes in string e.g. NDQ_018444 in val
                else:
                    pred_opt = 0
                #print pred_opt
                if not is_test_data:
                    max_options +=1 # because last option was removed by read_options_files for questions_dir_forbid
                    correct_ans_mat = self.ndq_test_data.read_correct_ans_file(question_dir_path,max_options)
                    correct_ans_mat[np.where(correct_ans_mat == -1)] = 0
                    #print correct_ans_mat
                    corr_opt = np.where(correct_ans_mat == 1)[1][0]
                    if corr_opt == pred_opt:
                        ndq_acc_pred+=1
                ndq_total_pred+=1
                quest_ans_dict[question_dir] = options_list[pred_opt]
                quest_mcq_dict[question_dir] = options_list[pred_opt]
                quest_tf_dict[question_dir] = "h"
            
            for question_dir in questions_dir_tf:
                flag = False
                question_dir_path = os.path.join(l_dir, question_dir)
                #print ("Question : ", question_dir)
                try :
                    options_mat,max_options = self.ndq_test_data.read_options_files(question_dir_path)
                    question_mat = self.ndq_test_data.read_question_file(question_dir_path)
                    sent_mat = self.ndq_test_data.read_sentence_file(question_dir_path)
                except:
                    total_missed_ques+=1
                    flag = True
                    print ("Lesson : ", lesson)
                    print ("Question : ", question_dir)
                    print "ERROR! THIS QUESTION HAS ERROR"
                
                # print "max option",max_options
                if max_options == 1 :
                    print ("max option is 1")
                    max_options = self.ndq_test_data.num_of_options_for_quest - 1
                pred_options_arr = ndq_model.predict([question_mat, sent_mat, options_mat])

                #Get maximum only from specified options not from complete list
                if flag==False:
                    pred_opt = np.argmax(pred_options_arr[0,:max_options])
                else:
                    pred_opt = 0

                if not is_test_data:
                    correct_ans_mat = self.ndq_test_data.read_correct_ans_file(question_dir_path,max_options)
                    correct_ans_mat[np.where(correct_ans_mat == -1)] = 0
                    corr_opt = np.where(correct_ans_mat == 1)[1][0]
                    if corr_opt == pred_opt:
                        ndq_acc_pred+=1
                ndq_total_pred+=1
                quest_ans_dict[question_dir] = options_list[pred_opt]
                quest_tf_dict[question_dir] = options_list[pred_opt]
                quest_mcq_dict[question_dir] = "c"
            
        print "NDQ Total questions : ", ndq_total_pred
        print "Accurate : ", ndq_acc_pred
        print "Total questions with error",total_missed_ques
        self.generate_result_file(quest_tf_dict,is_test_data)

    def generate_result_file(self,quest_ans_dict,is_test_data):

        total_un_ques = 0

        res_file_path = os.path.join(os.path.dirname(self.ndq_test_data.processed_data_path),"tqa_test-tf.json")
        res_f_handle = open(res_file_path, "w")

        json_path = os.path.dirname(os.path.dirname(self.ndq_test_data.processed_data_path))
        read_val_json = read_json(json_path,is_test_data)

        q_ids_list = read_val_json.get_questions_id()

        for q_id in q_ids_list:
            if quest_ans_dict.get(q_id,None) is None:
                #print "Question not present",q_id
                quest_ans_dict[q_id] = 'a'
                total_un_ques+=1

        print "Total untracked questions",total_un_ques
        ques_ans_json_data = json.dumps(quest_ans_dict, indent=4)
        res_f_handle.write(ques_ans_json_data)
        res_f_handle.close()

if __name__ == "__main__":
    from tqa_system import tqa_system
    train_data_path = "../data/train/processed_data/one_hot_files"
    val_data_path = "../data/val/processed_data/one_hot_files"
    test_data_path = "../data/test/processed_data/one_hot_files"
    tqa_sys = tqa_system(train_data_path,val_data_path,test_data_path)
    ndq_model,ndq_test_data = tqa_sys.train_ndq_model() # load the model with trained weights in tqa_system.py
    get_result = generate_result(ndq_test_data)
    get_result.predict_options_one_by_one(ndq_model, True)