import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import sys
import pickle
from train import NormaliseList

'''
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


'''

def decode_sequence(input_seq,encoder_model,decoder_model,num_decoder_tokens,targetVocabulary):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = targetVocabulary.to_token(sampled_token_index)
        decoded_sentence.append(sampled_char)

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == "EOS" or len(decoded_sentence) > num_decoder_tokens:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
        states_value = [h, c]
    return decoded_sentence




  
def Revert_back(X,dict_):
    temp = []
    for element in X:
        if element in dict_.keys():
            temp.append(dict_[element])
        elif element == "OOV":
            temp.append("OOV_Token")
        else :
            temp.append(element)
        
    return temp  
        



with open("Dictionaries.pkl",'rb') as f:
    vocab = pickle.load(f)
srcVocabulary = vocab[0]
targetVocabulary = vocab[1]
    

latent_dim = 32
batch_size = 64

num_encoder_tokens = 304 #len(input_characters)  # 250
num_decoder_tokens = 304 #len(target_characters)  # 250
max_encoder_seq_length = 37 #max([len(txt) for txt in input_texts])  # 22
max_decoder_seq_length = 37 #max([len(txt) for txt in target_texts]) # 22

#print("Number of samples:", len(X_train))
print("Number of unique input tokens(src vocab):", srcVocabulary.unique_tokens())
print("Number of unique output tokens(target vocab):", targetVocabulary.unique_tokens())
print("Max sequence length for inputs:", 37)
print("Max sequence length for outputs:", 37)


# one-hot encoder

valid_csv = sys.argv[1]
output_csv = sys.argv[2]
#valid_csv = "Valid.csv"
#output_csv = "Output.csv"

Valid = pd.read_csv(valid_csv)


#print(codeToken)

codeToken = Valid['sourceTokens'].values  # tokenized incorrect code
srcToken = list(codeToken)


# Normalised code tokens
X_val, dict_val_x = NormaliseList(Valid,"sourceLineTokens",srcToken)
#Y_val, dict_val_y = NormaliseList(Valid,"targetLineTokens",targetToken)


Valid_X = []
#Valid_Y = []
embedding_length = 35
for index,tokenList in enumerate(X_val):
    Valid_X.append(srcVocabulary.create_embedding(tokenList,embedding_length))

    
    
encoder_input_data = np.zeros(
    (len(Valid_X), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
)


for i, input_text in enumerate(Valid_X):
    for t, token in enumerate(input_text):
        encoder_input_data[i, t, token] = 1.0

        	
        	
# Define sampling models
# Restore the model and construct the encoder and decoder.
model = keras.models.load_model("s2s")

latent_dim = 32

encoder_inputs = model.input[0]  # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = keras.Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]  # input_2
decoder_state_input_h = keras.Input(shape=(latent_dim,), name="input_3")
decoder_state_input_c = keras.Input(shape=(latent_dim,), name="input_5")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = keras.Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)



   
#reader = csv.reader(open(valid_csv, 'rb'))
#reader1 = csv.reader(open('output1.csv', 'rb'))
#writer = csv.writer(open(output_csv, 'wb'))
#writer.writerow(["Unnamed:0","sourceText","targetText","sourceLineText","targetLineText","lineNums_Text","sourceTokens","targetTokns","sourceLineTokens","taretLineTokens","fixedTokens"])
    
write_Back = []
print("Prediction started!...")   
for seq_index in range(len(Valid_X)):
    # Take one sequence (part of the training set)model = keras.models.load_model("s2s")
    input_seq = encoder_input_data[seq_index : seq_index + 1]
    #decoded_sentence = Revert_back(decode_sequence(input_seq),dict_y[seq_index],seq_index)
    decoded_sentence = Revert_back(decode_sequence(input_seq,encoder_model,decoder_model,num_decoder_tokens,targetVocabulary),dict_val_x[seq_index])
    write_Back.append((Revert_back(decoded_sentence[:-1],dict_val_x[seq_index])))

Valid["fixedTokens"] = write_Back   
Valid.to_csv(output_csv)

print("Written in .csv file successfully!")



    




        
        

