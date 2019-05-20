import os
import numpy as np
from model import sciq_model
from data_prepare import prepare_data
from keras.callbacks import ModelCheckpoint

class sciq_system():
    def __init__(self,word2vec_path,train_data_path,val_data_path,test_data_path):
        self.word_vec_size = 300
        self.max_q_length = 68
        self.max_option_length = 12       
        self.max_opt_count = 4
        self.max_sent_para = 10
        self.max_words_sent = 25
        self.nb_epoch = 50
        self.steps_per_epoch = 117
        self.validation_steps = 10
        self.test_steps = 10
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.word2vec_path = word2vec_path
        self.models_path = os.path.join("../data/train","saved_models")
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)

    def train_model(self):
        #model_load_weights_fname = "cnn_model2_initial_weights.h5"
        model_fname = "cnn_model2_weights.h5"
        read_train_data = prepare_data(self.word2vec_path,self.train_data_path,self.word_vec_size,self.max_q_length,self.max_option_length,self.max_opt_count,self.max_sent_para,self.max_words_sent)
        read_val_data = prepare_data(self.word_vec_size, self.val_data_path, self.word_vec_size, self.max_q_length, 
                                       self.max_option_length, self.max_opt_count, self.max_sent_para, self.max_words_sent)
        read_test_data = prepare_data(self.word2vec_path, self.test_data_path, self.word_vec_size, self.max_q_length, 
                                       self.max_option_length, self.max_opt_count, self.max_sent_para, self.max_words_sent)
        model = sciq_model(self.word_vec_size,self.max_q_length,self.max_option_length,self.max_opt_count,self.max_sent_para,self.max_words_sent)
        train_model = model.get_cnn_model2()
        #train_model.load_weights(os.path.join(self.models_path,model_load_weights_fname))
        checkpointer = ModelCheckpoint(filepath=os.path.join(self.models_path,model_fname), verbose=1, save_best_only=True, save_weights_only=True)
        train_model.fit_generator(read_train_data.read_all_vectors(),steps_per_epoch=self.steps_per_epoch,epochs = self.nb_epoch,validation_data=read_val_data.read_all_vectors(),callbacks = [checkpointer], validation_steps=self.validation_steps,verbose=1)
        train_model.save_weights(os.path.join(self.models_path,model_fname))
        s1 = train_model.evaluate_generator(read_val_data.read_all_vectors(),steps=self.validation_steps)
        s2 = train_model.evaluate_generator(read_train_data.read_all_vectors(),steps=self.steps_per_epoch)
        #s3 = train_model.evaluate_generator(read_test_data.read_all_vectors(),steps=self.test_steps) # For testing on test set
        print(s1)
        print(s2)
        #print(s3)
        return train_model, read_val_data


if __name__ == "__main__":
    train_data_path = "../data/train/processed_data/one_hot_files"
    val_data_path = "../data/valid/processed_data/one_hot_files"
    test_data_path = "../data/test/processed_data/one_hot_files"
    word2vec_path = "../../word2vec/GoogleNews-vectors-negative300.bin.gz"
    tqa_sys = sciq_system(word2vec_path,train_data_path,val_data_path,test_data_path)
    tqa_sys.train_model()











