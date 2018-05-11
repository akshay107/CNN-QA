import os
import numpy as np
from model import tqa_model
from data_prepare import prepare_data
from result import generate_result

class tqa_system():
    def __init__(self,word2vec_path,train_data_path,val_data_path,test_data_path):
        self.word_vec_size = 300
        self.max_q_length = 65
        self.max_option_length = 25      
        self.max_opt_count = 7
        self.max_sent_para = 10
        self.max_words_sent = 20
        self.nb_epoch = 50
        self.batch_size = 16
        self.steps_per_epoch_dq = 54
        self.validation_steps_dq = 16
        self.steps_per_epoch_ndq = 333
        self.validation_steps_ndq = 100
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.word2vec_path = word2vec_path
        self.models_path = os.path.join("../data/train","saved_models")
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)

    def train_ndq_model(self):
        #model_load_weights_fname = "cnn_model2_initial_weights.h5"
        model_fname = "cnn_model2_weights.h5"
        read_train_data = prepare_data(self.word2vec_path,self.train_data_path,False,self.word_vec_size,self.max_q_length,self.max_option_length,self.max_opt_count,self.max_sent_para,self.max_words_sent)
        read_val_data = prepare_data(self.word2vec_path, self.val_data_path,False, self.word_vec_size, self.max_q_length,
                                       self.max_option_length, self.max_opt_count, self.max_sent_para, self.max_words_sent)
        read_test_data = prepare_data(self.word2vec_path, self.test_data_path,True, self.word_vec_size, self.max_q_length,
                                       self.max_option_length, self.max_opt_count, self.max_sent_para, self.max_words_sent)
        model = tqa_model(self.word_vec_size,self.max_q_length,self.max_option_length,self.max_opt_count,self.max_sent_para,self.max_words_sent)
        train_model = model.get_cnn_model2()
        #train_model.load_weights(os.path.join(self.models_path,model_load_weights_fname))
        train_model.fit_generator(read_train_data.read_all_vectors_for_ndq(),steps_per_epoch=self.steps_per_epoch_ndq,epochs = self.nb_epoch,validation_data=read_val_data.read_all_vectors_for_ndq(),validation_steps=self.validation_steps_ndq,verbose=1)
        #train_model.save_weights(os.path.join(self.models_path,model_fname))
        s1 = train_model.evaluate_generator(read_val_data.read_all_vectors_for_ndq(),steps=self.validation_steps_ndq)
        s2 = train_model.evaluate_generator(read_train_data.read_all_vectors_for_ndq(),steps=self.steps_per_epoch_ndq)
        print(s1)
        print(s2)      
        return train_model, read_test_data


    def generate_result(self):
        ndq_model,ndq_test_data = self.train_ndq_model()
        get_result = generate_result(ndq_test_data)
        get_result.predict_options_one_by_one(ndq_model, True) # True for test data, False otherwise



if __name__ == "__main__":
    '''train_data_path = "../data/train/processed_data/one_hot_files"
    val_data_path = "../data/val/processed_data/one_hot_files"
    test_data_path = "../data/test/processed_data/one_hot_files"'''
    train_data_path = "/home/cvpr/akshay/TQA/train/processed_data/one_hot_files"
    val_data_path = "/home/cvpr/akshay/TQA/val/processed_data/one_hot_files"
    test_data_path = "/home/cvpr/akshay/TQA/test/processed_data/one_hot_files"
    word2vec_path = "../../word2vec/GoogleNews-vectors-negative300.bin.gz"
    tqa_sys = tqa_system(word2vec_path,train_data_path,val_data_path,test_data_path)
    tqa_sys.train_ndq_model()










