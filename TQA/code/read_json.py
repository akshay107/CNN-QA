import json
import os
import shutil
import nltk
from nltk.tokenize import word_tokenize
from get_closest_sen import get_closest_sentences

class read_json():
    def __init__(self,json_dir,is_test_data,op_dir = None):
        self.is_test_data = is_test_data
        self.json_dir = json_dir
        if not op_dir:
            op_dir = os.path.join(json_dir,"processed_data","text_question_sep_files")
        if not os.path.exists(op_dir):
            os.makedirs(op_dir)
        self.op_dir = op_dir
        self.json_file_list = self.get_list_of_files(self.json_dir,file_extension=".json")

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

        l_id_tag = 'globalID'
        qs_tag = 'questions'
        img_path_tag = "imagePath"
        ndq_tag = 'nonDiagramQuestions'
        dq_tag = "diagramQuestions"
        q_tags = [ndq_tag,dq_tag]

        for f in self.json_file_list:
            with open(os.path.join(self.json_dir,f), 'r') as f:
                data = json.load(f)
            for lessons in data:
                lesson_id = lessons[l_id_tag]
                print "Lesson : ",lesson_id
                l_dir = os.path.join(self.op_dir,lesson_id)
                if not os.path.exists(l_dir):
                    os.makedirs(l_dir)
                topic_f = "topics.txt"
                topic_f_handle = open(os.path.join(l_dir,topic_f), 'w')
                for topics in lessons['topics']:
                    topic_f_handle.write(lessons['topics'][topics]['content']['text'] + "\n")
                topic_f_handle.close()
                for q_tag in q_tags:
                    # if lessons[qs_tag][q_tag]
                    for q_id,_ in lessons[qs_tag][q_tag].iteritems():
                        q_dir = os.path.join(l_dir,str(q_id))
                        os.makedirs(q_dir)
                        question = lessons[qs_tag][q_tag][q_id]['beingAsked']['processedText']
                        question = question.encode('ascii', 'ignore').decode('ascii')
                        question_f = "Question"+ ".txt"
                        question_f_handle = open(os.path.join(q_dir, question_f), 'w')
                        question_f_handle.write(question+"\n")
                        question_f_handle.close()

                        option_list = []
                        option = 'a'
                        while lessons[qs_tag][q_tag][q_id]['answerChoices'].get(option,None) is not None:
                            option_list.append(lessons[qs_tag][q_tag][q_id]['answerChoices'][option]['processedText'])
                            option = chr(ord(option) + 1)
                        option = 'a'
                        for opt in option_list:
                            option_f = option+".txt"
                            opt = opt.encode('ascii', 'ignore').decode('ascii')
                            option_f_handle = open(os.path.join(q_dir, option_f), 'w')
                            option_f_handle.write(opt)
                            option_f_handle.close()
                            option = chr(ord(option) + 1)

                        if q_tag == dq_tag:
                            quest_img_path = os.path.join(self.json_dir,lessons[qs_tag][q_tag][q_id][img_path_tag])
                            shutil.copy2(quest_img_path, q_dir)

                        if not self.is_test_data:
                            correct_answer = lessons[qs_tag][q_tag][q_id]['correctAnswer']['processedText']
                            corr_f_handle = open(os.path.join(q_dir, "correct_answer.txt"), 'w')
                            corr_f_handle.write(correct_answer + "\n")
                            corr_f_handle.close()

    def get_questions_id(self):

        l_id_tag = 'globalID'
        qs_tag = 'questions'
        ndq_tag = 'nonDiagramQuestions'
        dq_tag = "diagramQuestions"
        q_tags = [ndq_tag, dq_tag]
        q_ids_list = []
        for f in self.json_file_list:
            with open(os.path.join(self.json_dir,f), 'r') as f:
                data = json.load(f)
            for lessons in data:
                for q_tag in q_tags:
                    for q_id,_ in lessons[qs_tag][q_tag].iteritems():
                        q_ids_list.append(q_id)

        return q_ids_list




    def sanity_test(self):

        print 20 * "*"
        print "DOING SANITY TESTING"

        lessons = self.get_list_of_dirs(self.op_dir)
        que_fname = "Question"
        corr_ans_fname = "correct_answer"
        closest_sent_fname = "closest_sent"
        f_ext = ".txt"
        num_of_ques = 0
        wrong_que = 0
        for lesson in lessons:
            print "Lesson : ",lesson
            questions_list = self.get_list_of_dirs(os.path.join(self.op_dir,lesson))
            for que in questions_list:
                num_of_ques +=1
                print "Question : ",que
                que_dir_path = os.path.join(self.op_dir,lesson,que)
                file_list = self.get_list_of_files(que_dir_path,file_extension=".txt")
                if que_fname+f_ext not in file_list :
                    print "Question file doesn't exist"
                if corr_ans_fname+f_ext not in file_list:
                    print "Correct answer file doesn't exist"
                if que.startswith('NDQ') and closest_sent_fname+f_ext not in file_list:
                    print "Closest sentence file doesn't exist"
                if not self.is_test_data:
                    with open(os.path.join(que_dir_path, corr_ans_fname+f_ext),"r") as f:
                        correct_answer = f.readlines()
                    correct_answer = correct_answer[0].strip().lower()
                    option = 'a'
                    while ord(option) <= ord(correct_answer):
                        if option+f_ext not in file_list:
                            wrong_que+=1
                            print "Correct answer is : ", correct_answer
                            print "Error : Option file doesn't exist",option+f_ext
                            shutil.rmtree(que_dir_path)
                            break
                        option = chr(ord(option) + 1)
                else:
                    print "cant check correctness of options as this is test data"
            print 20*"**"
        print "Total Questions : ",num_of_ques
        print "Questions with error : ",wrong_que

    def get_statistics(self):

        print 20 * "*"
        print "GETIING STATISTICS FOR THE DATA"

        lessons = self.get_list_of_dirs(self.op_dir)
        que_fname = "Question"
        corr_ans_fname = "correct_answer"
        closest_sent_fname = "closest_sent"
        f_ext = ".txt"
        num_que_token_list = []
        num_opt_token_list = []
        num_sent_token_list = []
        for lesson in lessons:
            questions_list = self.get_list_of_dirs(os.path.join(self.op_dir,lesson))
            for que in questions_list:
                que_dir_path = os.path.join(self.op_dir,lesson,que)
                file_list = self.get_list_of_files(que_dir_path,file_extension=".txt")
                with open(os.path.join(que_dir_path, que_fname+f_ext),"r") as f:
                    ques = f.readlines()
                num_of_tokens_in_que = 0
                for sent in ques:
                    words = word_tokenize(sent)
                    num_of_tokens_in_que+=len(words)
                num_que_token_list.append(num_of_tokens_in_que)

                if que.startswith('NDQ'):
                    with open(os.path.join(que_dir_path, closest_sent_fname+f_ext),"r") as f:
                        sents = f.readlines()
                    num_of_tokens_in_sents = 0
                    for sent in sents:
                        words = word_tokenize(sent)
                        num_of_tokens_in_sents+=len(words)
                    num_sent_token_list.append(num_of_tokens_in_sents)
                    if num_of_tokens_in_sents == 0:
                        print "Lesson : ",lesson
                        print "Question : ",que

                option = 'a'
                while os.path.exists(os.path.join(que_dir_path,option + f_ext)):
                    with open(os.path.join(que_dir_path, option + f_ext), "r") as f:
                        opt = f.readlines()
                    num_of_tokens_in_opt = 0
                    for sent in opt:
                        words = word_tokenize(sent)
                        num_of_tokens_in_opt += len(words)
                    num_opt_token_list.append(num_of_tokens_in_opt)
                    option = chr(ord(option) + 1)

        que_lenth_dict = nltk.FreqDist(x for x in num_que_token_list)
        print "Question length info"
        for k, v in que_lenth_dict.most_common(50):
            print str(k), str(v)
        print "Max question length : ",max(num_que_token_list)

        opt_lenth_dict = nltk.FreqDist(x for x in num_opt_token_list)
        print "Option length info"
        for k, v in opt_lenth_dict.most_common(50):
            print str(k), str(v)
        print "Max Option length : ", max(num_opt_token_list)

        sent_lenth_dict = nltk.FreqDist(x for x in num_sent_token_list)
        print "Closest sentence info"
        for k, v in sent_lenth_dict.most_common(1000):
            print str(k), str(v)
        print "Max Closest sentence length : ", max(num_sent_token_list)


    def read_json_do_sanity_create_closest_sent_(self):
        self.read_content()
        processed_data_path = os.path.dirname(self.op_dir)
        read_training_json = get_closest_sentences(processed_data_path)
        read_training_json.generate_closest_sentence()
        self.get_statistics()
        self.sanity_test()
