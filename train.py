import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pickle


class Vocabulary:

    def __init__(self, name):
        PAD_token = 0   # Used for padding short sentences
        SOS_token = 1   # Start-of-sentence token
        EOS_token = 2   # End-of-sentence token
        OOV_token = 3
        self.name = name
        self.token2index = {}
        self.token2count = {}
        self.index2token = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", OOV_token: "OOV"}
        self.sorted_dict = {}
        self.num_tokens = 4  # unique tokens
        self.num_sentences = 0
        self.longest_sentence = 0
        self.top_500 = []
        

    def add_token(self, token):
        if token not in self.token2index:
            # First entry of token into vocabulary
            self.token2index[token] = self.num_tokens
            self.token2count[token] = 1
            self.index2token[self.num_tokens] = token
            self.num_tokens += 1  # increase number of unique tokens
        else:
            # token exists; increase token count
            self.token2count[token] += 1
            
    def add_sentence(self, sentence):
        sentence_len = 0
        for token in sentence:
            sentence_len += 1
            self.add_token(token)
        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1

    def to_token(self, index):
        return self.index2token[index]

    def to_index(self, token):
        return self.token2index[token]
    
    def print_longest_sentence_length(self):
        print(f"Longest sentence = {self.longest_sentence}, #sentences = {self.num_sentences}")
    
    def top_k(self):
        self.sorted_dict = {key: val for key,val in sorted(self.token2count.items(), key=lambda item: item[1], reverse = True)}
        for k,v in self.sorted_dict.items():
            self.top_500.append(k)
            
        # generate new mapping
        self.index2token.clear()
        self.token2index.clear()
        self.index2token = {0: "PAD", 1: "SOS", 2: "EOS", 3: "OOV"}
        self.token2index = {"PAD": 0, "SOS":1, "EOS":2, "OOV": 3}
        
        cnt = 4
        for element in self.top_500:
            self.token2index[element] = cnt
            cnt += 1
            
        for k,v in self.token2index.items():
            self.index2token[v] = k

    
    def unique_tokens(self):
        return len(self.token2index.keys())
        
    
    
    def postprocess_embedding(self,tokenList,length):
        tokenList.insert(0,1)
        if len(tokenList) > length+1:
            return tokenList[:length+1] + [2]
        elif len(tokenList) < length+1:
            tokenList.append(2)
            for i in range(len(tokenList)-1,length+1):
                tokenList.append(0)  # PAD appended
            return tokenList
        return tokenList + [2]
    
    def create_embedding(self,tokenList,embedding_length):
        embedded_list = []
        for element in tokenList:
            if element in self.top_500[:300]:
                embedded_list.append(self.token2index[element])
            else:
                embedded_list.append(3)  # index for OOV_token
        return self.postprocess_embedding(embedded_list,embedding_length)





def NormaliseList(Data,_name,Tokens):
    X = [] # empty list for tokenised code
    list_of_dictionaries = []

    CodeToken = Data[_name].values
    CodeToken = list(CodeToken)


    dataTypes = ["int","char","double","float"]
    invalid = ["main",';',':','(',')','{','}','[',']']

    
    
    for codeNum,code in enumerate(Tokens): # iterate over codes
        #print(code)
        Dict = {"int":[],"float":[],"char":[],"double":[]}
        VAR_normalisation = {}
        
        Code = eval(code)

        for line in Code:                   # iterate over lines of a single code
            for index in range(len(line)):  # iterate over words/tokens of a single line of a code
                word = line[index]
                if word in dataTypes:
                    for pos in range(index+1,len(line),1):
                        if line[pos] in invalid:
                            break
                        elif line[pos] != ',' and not line[pos].isnumeric() and not (len(line[pos])>=2 and line[pos][0] == '"' and line[pos][-1] == '"'):
                            Dict[word].append(line[pos])
                
                
        
        #list_of_dictionaries.append(VAR_normalisation)

#         temp = []
#         for token in eval(CodeToken[codeNum]):
#             if token in tempDict.keys():
#                 temp.append(tempDict[token])
#             else:
#                 temp.append(token)
#         X.append(temp)
        
        var_int = 0
        var_char = 0
        var_float = 0
        var_double = 0
        #VAR_normalisation
        temp = []
        
        for token in eval(CodeToken[codeNum]):
            temp.append(token)
            for dType in dataTypes:
                for values in Dict[dType]:
                    if values == token:
                        
                        del temp[-1]
                        var_int += int(dType == "int")
                        var_char += int(dType == "char")
                        var_float += int(dType == "float")
                        var_double += int(dType == "double")
                        
                        if dType == 'int':
                            VAR_normalisation["VAR_" + dType + '_' + str(var_int)] = token
                            temp.append("VAR_" + dType + '_' + str(var_int))
                        if dType == 'char':
                            VAR_normalisation["VAR_" + dType + '_' + str(var_char)] = token
                            temp.append("VAR_" + dType + '_' + str(var_char))
                        if dType == 'float':
                            VAR_normalisation["VAR_" + dType + '_' + str(var_float)] = token
                            temp.append("VAR_" + dType + '_' + str(var_float))
                        if dType == 'double':
                            VAR_normalisation["VAR_" + dType + '_' + str(var_double)] = token
                            temp.append("VAR_" + dType + '_' + str(var_double))
                            
                        break
                        
        
        X.append(temp)                         
        list_of_dictionaries.append(VAR_normalisation)  

    return X,list_of_dictionaries




if __name__=='__main__':
    Data = pd.read_csv("train.csv")
    
    
    
    codeToken = Data['targetTokens'].values  #  tokenized correct code
    targetToken = list(codeToken)
    #print(codeToken)
    
    codeToken = Data['sourceTokens'].values  # tokenized incorrect code
    srcToken = list(codeToken)
    
    
    # Normalised code tokens
    X, dict_x = NormaliseList(Data,"sourceLineTokens",srcToken)
    Y, dict_y = NormaliseList(Data,"targetLineTokens",targetToken)
    
    # creating separate vocabularies for source and target code token lists
    from train import Vocabulary
    vocab = []
    srcVocabulary = Vocabulary("Assignment-2-src")
    targetVocabulary = Vocabulary("Assignment-2-target")
    
    
    for index,tokenList in enumerate(X):
    	srcVocabulary.add_sentence(tokenList)
        
    srcVocabulary.print_longest_sentence_length()
    srcVocabulary.top_k()
    #print(top_250,top_500,top_1000)
    
    for index,tokenList in enumerate(Y):
    	targetVocabulary.add_sentence(tokenList)
            
    targetVocabulary.print_longest_sentence_length()
    targetVocabulary.top_k()
    
    vocab.append(srcVocabulary)
    vocab.append(targetVocabulary)
    
    # store vocabularies
    with open("Dictionaries.pkl",'wb') as f:
    	pickle.dump(vocab,f,pickle.HIGHEST_PROTOCOL)
    # creating separate vocabularies for source and target code token lists
    
    
    X_train = []
    Y_train = []
    embedding_length = 35
    for index,tokenList in enumerate(X):
    	X_train.append(srcVocabulary.create_embedding(tokenList,embedding_length))
    
    for index,tokenList in enumerate(Y):
    	Y_train.append(targetVocabulary.create_embedding(tokenList,embedding_length))
        
        
    latent_dim = 32
    batch_size = 64
    epochs = 40
    
    #input_characters = sorted(list(input_characters))
    #target_characters = sorted(list(target_characters))
    num_encoder_tokens = 304 #len(input_characters)  # 250
    num_decoder_tokens = 304 #len(target_characters)  # 250
    max_encoder_seq_length = 37 #max([len(txt) for txt in input_texts])  # 22
    max_decoder_seq_length = 37 #max([len(txt) for txt in target_texts]) # 22
    
    print("Number of samples:", len(X_train))
    print("Number of unique input tokens:", srcVocabulary.unique_tokens())
    print("Number of unique output tokens:", targetVocabulary.unique_tokens())
    print("Max sequence length for inputs:", 37)
    print("Max sequence length for outputs:", 37)
    
    #input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    #target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
    
    # one-hot encoder
    
    encoder_input_data = np.zeros(
        (len(X_train), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
    )
    decoder_input_data = np.zeros(
        (len(X_train), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
    )
    decoder_target_data = np.zeros(
        (len(X_train), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
    )
    
    
    for i, (input_text, target_text) in enumerate(zip(X_train, Y_train)):
    	for t, token in enumerate(input_text):
    		encoder_input_data[i, t, token] = 1.0
    	for t, token in enumerate(target_text):
    		decoder_input_data[i, t, token] = 1.0
    		if t > 0:
    			decoder_target_data[i, t - 1, token] = 1.0
    
    """
    ## Build the model
    """
    
    # Define an input sequence and process it.
    encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
    encoder = keras.layers.LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]
    
    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))
    
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    """
    ## Train the model
    """
    
    model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
    )
    # Save model
    model.save("s2s")
    print("Training Complete and Model Saved !")
    """
    ## Run inference (sampling)
    1. encode input and retrieve initial decoder state
    2. run one step of decoder with this initial state
    and a "start of sequence" token as target.
    Output will be the next target token.
    3. Repeat with the current target token and current states
    """







