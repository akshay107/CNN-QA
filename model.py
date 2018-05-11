# The code is written using Keras with Theano backend. 
# get_cnn_model1 refers to cnn_{3,4,5} and get_cnn_model2 refers to cnn_{2,3,4}


import numpy as np
import theano.tensor as T
from keras.models import Model
from keras.layers import Dense,TimeDistributed,RepeatVector,Input, LSTM, GRU, Merge, Lambda, Masking,Reshape, Activation,Conv2D
from keras.layers.merge import Concatenate, Dot, Add
from keras import initializers,regularizers,constraints
from keras.engine.topology import Layer
from keras.optimizers import SGD
import keras.backend as K

class MaskedSoftmax(Layer):
    # This layers ensures zero probability for zero padded option vectors
    def build(self, input_shape):
        assert len(input_shape[0]) == 2
        assert len(input_shape[1]) == 4
    def call(self,inputs):
        arr1 = inputs[0]
        arr2 = inputs[1]
        arr3 = arr2.norm(2,axis=3)
        arr4 = arr3.sum(axis=2)
        x_mask = T.switch(T.eq(arr4,0),np.NINF,arr1)
        sm = T.nnet.softmax(x_mask)
        return sm
    def compute_output_shape(self,input_shape):
        return input_shape[0]

class sciq_model():
    def __init__(self,word_vec_size,max_q_length,max_option_length,max_opt_count,max_sent_para,max_words_sent):
        self.max_q_length = max_q_length # maximum question length
        self.max_option_length = max_option_length # maximum option length
        self.max_opt_count = max_opt_count # maximum number of options
        self.max_sent_para = max_sent_para # maximum number of sentences in the paragraph
        self.max_words_sent = max_words_sent # maximum number of words in the sentence of the paragraph
        self.word_vec_size = word_vec_size

    def get_gru_baseline(self):
        lstm_qo = GRU(100,return_sequences=False)
        get_diag = Lambda(lambda xin: K.sum(xin*T.eye(self.max_opt_count),axis=2),output_shape=(self.max_opt_count,))
        transp_out = Lambda(lambda xin: K.permute_dimensions(xin,(0,2,1)),output_shape=(self.max_opt_count,100))
        apply_weights = Lambda(lambda xin: (K.expand_dims(xin[0],axis=-1)*K.expand_dims(xin[1],axis=2)).sum(axis=1), output_shape=(100,self.max_opt_count))
        tile_q = Lambda(lambda xin: K.tile(xin,(1,self.max_opt_count,1,1)),output_shape=(self.max_opt_count,self.max_q_length,self.word_vec_size))
        exp_dims = Lambda(lambda xin: K.expand_dims(xin,1), output_shape=(1,self.max_q_length,self.word_vec_size))
        exp_layer = Lambda(lambda xin: K.exp(xin), output_shape=(self.max_sent_para,self.max_opt_count))
        mask_weights = Lambda(lambda xin: T.switch(T.eq(xin,0),np.NINF,xin), output_shape=(self.max_sent_para,self.max_opt_count))
        final_weights = Lambda(lambda xin: xin/K.cast(K.sum(xin, axis=1, keepdims=True), K.floatx()),output_shape=(self.max_sent_para,self.max_opt_count))


        q_input = Input(shape=(self.max_q_length, self.word_vec_size), name='question_input')
        q_exp = exp_dims(q_input)
        q_rep = tile_q(q_exp)
        option_input = Input(shape=(self.max_opt_count, self.max_option_length,self.word_vec_size), name='option_input')
        opt_q = Concatenate(axis=2)([q_rep,option_input])

        lstm_input = Input(shape=(None, self.word_vec_size), name='lstm_input')
        lstm_mask = Masking(mask_value=0.)(lstm_input)
        lstm_out = lstm_qo(lstm_mask)

        lstm_model = Model(inputs=lstm_input,outputs=lstm_out)
        lstm_td_opt = TimeDistributed(lstm_model)(opt_q)
        
        doc_input = Input(shape=(self.max_sent_para, self.max_words_sent, self.word_vec_size), name='doc_input')
        lstm_doc = TimeDistributed(lstm_model)(doc_input)
        att_wts = Dot(axes=2,normalize=True)([lstm_doc,lstm_td_opt])
        att_wts = mask_weights(att_wts)
        att_wts = exp_layer(att_wts)
        att_wts = final_weights(att_wts)
        out = apply_weights([lstm_doc,att_wts])

        out = transp_out(out)
        dp = Dot(axes=2,normalize=True)([out,lstm_td_opt])
        out = get_diag(dp)
        probs = MaskedSoftmax()([out,option_input])
        main_model = Model(inputs=[q_input,doc_input,option_input],outputs=probs)
        sgd = SGD(lr=0.1, decay=0., momentum=0., nesterov=False)
        main_model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
        main_model.summary()
        return main_model

    def get_cnn_model1(self):
        get_diag = Lambda(lambda xin: K.sum(xin*T.eye(self.max_opt_count),axis=2),output_shape=(self.max_opt_count,))
        transp_out = Lambda(lambda xin: K.permute_dimensions(xin,(0,2,1)),output_shape=(self.max_opt_count,self.word_vec_size))
        apply_weights = Lambda(lambda xin: (K.expand_dims(xin[0],axis=-1)*K.expand_dims(xin[1],axis=2)).sum(axis=1), output_shape=(self.word_vec_size,self.max_opt_count))
        tile_q = Lambda(lambda xin: K.tile(xin,(1,self.max_opt_count,1,1)),output_shape=(self.max_opt_count,self.max_q_length,self.word_vec_size))
        exp_dims = Lambda(lambda xin: K.expand_dims(xin,1), output_shape=(1,self.max_q_length,self.word_vec_size))
        exp_dims2 = Lambda(lambda xin: K.expand_dims(xin,3), output_shape=(None,self.word_vec_size,1))
        exp_layer = Lambda(lambda xin: K.exp(xin), output_shape=(self.max_sent_para,self.max_opt_count))
        final_weights = Lambda(lambda xin: xin/K.cast(K.sum(xin, axis=1, keepdims=True), K.floatx()),output_shape=(self.max_sent_para,self.max_opt_count))
        mask_weights = Lambda(lambda xin: T.switch(T.eq(xin,0),np.NINF,xin), output_shape=(self.max_sent_para,self.max_opt_count))
        glob_pool = Lambda(lambda xin: K.mean(xin, axis=[1, 2]),output_shape=(100,))

        filter_sizes = [3,4,5]
        num_filters = 100
        q_input = Input(shape=(self.max_q_length, self.word_vec_size), name='question_input')
        q_exp = exp_dims(q_input)
        q_rep = tile_q(q_exp)
        option_input = Input(shape=(self.max_opt_count, self.max_option_length,self.word_vec_size), name='option_input')
        opt_q = Concatenate(axis=2)([q_rep,option_input])

        cnn_input = Input(shape=(None, self.word_vec_size), name='cnn_input')
        cnn_reshape = exp_dims2(cnn_input)

        conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], self.word_vec_size), padding='valid', kernel_initializer='normal', activation='linear')(cnn_reshape)
        conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], self.word_vec_size), padding='valid', kernel_initializer='normal', activation='linear')(cnn_reshape)
        conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], self.word_vec_size), padding='valid', kernel_initializer='normal', activation='linear')(cnn_reshape)

        meanpool_0 = glob_pool(conv_0)
        meanpool_1 = glob_pool(conv_1)
        meanpool_2 = glob_pool(conv_2)
        concatenated_tensor = Concatenate(axis=1)([meanpool_0, meanpool_1, meanpool_2])

        cnn_model = Model(inputs=cnn_input,outputs=concatenated_tensor)
        cnn_td_opt = TimeDistributed(cnn_model)(opt_q)
        
        doc_input = Input(shape=(self.max_sent_para, self.max_words_sent, self.word_vec_size), name='doc_input')
        cnn_doc = TimeDistributed(cnn_model)(doc_input)
        att_wts = Dot(axes=2,normalize=True)([cnn_doc,cnn_td_opt])
        att_wts = mask_weights(att_wts)
        att_wts = exp_layer(att_wts)
        att_wts = final_weights(att_wts)
        out = apply_weights([cnn_doc,att_wts])

        out = transp_out(out)
        dp = Dot(axes=2,normalize=True)([out,cnn_td_opt])
        out = get_diag(dp)
        probs = MaskedSoftmax()([out,option_input])
        main_model = Model(inputs=[q_input,doc_input,option_input],outputs=probs)
        sgd = SGD(lr=0.1, decay=0., momentum=0., nesterov=False)
        main_model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
        main_model.summary()
        return main_model

    def get_cnn_model2(self):
        get_diag = Lambda(lambda xin: K.sum(xin*T.eye(self.max_opt_count),axis=2),output_shape=(self.max_opt_count,))
        transp_out = Lambda(lambda xin: K.permute_dimensions(xin,(0,2,1)),output_shape=(self.max_opt_count,self.word_vec_size))
        apply_weights = Lambda(lambda xin: (K.expand_dims(xin[0],axis=-1)*K.expand_dims(xin[1],axis=2)).sum(axis=1), output_shape=(self.word_vec_size,self.max_opt_count))
        tile_q = Lambda(lambda xin: K.tile(xin,(1,self.max_opt_count,1,1)),output_shape=(self.max_opt_count,self.max_q_length,self.word_vec_size))
        exp_dims = Lambda(lambda xin: K.expand_dims(xin,1), output_shape=(1,self.max_q_length,self.word_vec_size))
        exp_dims2 = Lambda(lambda xin: K.expand_dims(xin,3), output_shape=(None,self.word_vec_size,1))
        exp_layer = Lambda(lambda xin: K.exp(xin), output_shape=(self.max_sent_para,self.max_opt_count))
        final_weights = Lambda(lambda xin: xin/K.cast(K.sum(xin, axis=1, keepdims=True), K.floatx()),output_shape=(self.max_sent_para,self.max_opt_count))
        mask_weights = Lambda(lambda xin: T.switch(T.eq(xin,0),np.NINF,xin), output_shape=(self.max_sent_para,self.max_opt_count))
        glob_pool = Lambda(lambda xin: K.mean(xin, axis=[1, 2]),output_shape=(100,))

        filter_sizes = [2,3,4]
        num_filters = 100
        q_input = Input(shape=(self.max_q_length, self.word_vec_size), name='question_input')
        q_exp = exp_dims(q_input)
        q_rep = tile_q(q_exp)
        option_input = Input(shape=(self.max_opt_count, self.max_option_length,self.word_vec_size), name='option_input')
        opt_q = Concatenate(axis=2)([q_rep,option_input])

        cnn_input = Input(shape=(None, self.word_vec_size), name='cnn_input')
        cnn_reshape = exp_dims2(cnn_input)

        conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], self.word_vec_size), padding='valid', kernel_initializer='normal', activation='linear')(cnn_reshape)
        conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], self.word_vec_size), padding='valid', kernel_initializer='normal', activation='linear')(cnn_reshape)
        conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], self.word_vec_size), padding='valid', kernel_initializer='normal', activation='linear')(cnn_reshape)

        meanpool_0 = glob_pool(conv_0)
        meanpool_1 = glob_pool(conv_1)
        meanpool_2 = glob_pool(conv_2)
        concatenated_tensor = Concatenate(axis=1)([meanpool_0, meanpool_1, meanpool_2])

        cnn_model = Model(inputs=cnn_input,outputs=concatenated_tensor)
        cnn_td_opt = TimeDistributed(cnn_model)(opt_q)
        
        doc_input = Input(shape=(self.max_sent_para, self.max_words_sent, self.word_vec_size), name='doc_input')
        cnn_doc = TimeDistributed(cnn_model)(doc_input)
        att_wts = Dot(axes=2,normalize=True)([cnn_doc,cnn_td_opt])
        att_wts = mask_weights(att_wts)
        att_wts = exp_layer(att_wts)
        att_wts = final_weights(att_wts)
        out = apply_weights([cnn_doc,att_wts])

        out = transp_out(out)
        dp = Dot(axes=2,normalize=True)([out,cnn_td_opt])
        out = get_diag(dp)
        probs = MaskedSoftmax()([out,option_input])
        main_model = Model(inputs=[q_input,doc_input,option_input],outputs=probs)
        sgd = SGD(lr=0.1, decay=0., momentum=0., nesterov=False)
        main_model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
        main_model.summary()
        return main_model