from keras.layers import Concatenate, Conv1D, Dense, Dropout, Embedding, Input, LeakyReLU, LSTM, Reshape
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras.regularizers import L1L2


def generate_model(feat_dict):

    con_feats = feat_dict['con_feats']
    lstm_feats = feat_dict['lstm_feats']
    M = feat_dict['M']

    # Initialize input
    INPUT = [[], [], []]

    # A) USER PROFILE FEATURE LAYERS
    # categorical features
    for i, m in enumerate(M):
        INPUT[0].append(Input(shape=(None, 1), name='cat_' + str(i) + '_input'))
        INPUT[1].append(Embedding(m[0], min(m), name='cat_' + str(i) + '_embedding')(INPUT[0][-1]))
        INPUT[2].append(Reshape((-1, min(m)), name='cat_' + str(i) + '_reshape')(INPUT[1][-1]))
    # continuous features
    cont_input = Input(shape=(None, len(con_feats)), name='cont_input')
    INPUT[2].append(cont_input)
    # input concatenation
    concat1 = Concatenate(name='profile_concat')(INPUT[2])

    # B) LSTM LAYERS
    lstm_input = Input(shape=(None, len(lstm_feats)), name='lstm_input')
    # lstm1 from input lstm_input
    lstm1 = LSTM(80, recurrent_regularizer=L1L2(), dropout=.1, return_sequences=True,
                 input_shape=(None, len(lstm_feats)), name='lstm_layer_1')(lstm_input)
    # lstm2 from input lstm1
    lstm2 = LSTM(32, recurrent_regularizer=L1L2(), dropout=.1, return_sequences=True,
                 name='lstm_layer_2')(lstm1)
    concat2 = Concatenate(name='profile_lstm_concat')(INPUT[2] + [lstm2])

    # C) DENSE LAYERS
    # dns1 from concat2
    dns1 = Dense(128 * 3, name='dense_layer_1')(concat2)
    dns1 = LeakyReLU()(dns1)
    concat3 = Dropout(.1)(dns1)
    # dns2 from dns1
    dns2 = Dense(128 * 1, name='dense_layer_2')(concat3)
    dns2 = LeakyReLU()(dns2)
    # drpt1 from dns2
    drpt1 = Dropout(.2)(dns2)
    # dns3 from drpt1
    dns3 = Dense(1, name='dense_layer_3')(drpt1)

    # MODEL
    model = Model(INPUT[0] + [cont_input, lstm_input], dns3)
    print(model.summary(line_length=80))
    rms_prop = RMSprop(lr=0.0001)
    model.compile(rms_prop,
                  loss='mean_squared_error',
                  metrics=['mean_squared_error', 'mean_absolute_error'],
                  sample_weight_mode='temporal') # timestep-wise sample weighting

    return model


def generate_model2(con_cols, lstm_list, M, cells):

    # Initialize input
    k = 0
    inputs = [[], [], []]
    for m in M:
        inputs[0].append(Input(shape=(None, 1)))
        inputs[1].append(Embedding(m[0], min(m[0], m[1]))(inputs[0][-1]))
        inputs[2].append(Reshape((-1, min(m[0], m[1])))(inputs[1][-1]))
        k += min(m[0], m[1])

    cont_input = Input(shape=(None, len(con_cols)), name='cont_input')
    inputs[2].append(cont_input)
    # input concatenation
    concat1 = Concatenate(name='profile_concat')(inputs[2])

    # LSTM layers
    lstm_input = Input(shape=(None, len(lstm_list)))
    k += len(lstm_list)
    lstm = LSTM(cells, return_sequences=True, input_shape=(None, k), stateful=False, dropout=0.1,
                recurrent_regularizer=L1L2(l1=0.0))(lstm_input)
    lstm1 = LSTM(12, return_sequences=True, input_shape=(None, k), stateful=False, dropout=0.1,
                recurrent_regularizer=L1L2(l1=0.0))(lstm)
    lstm2 = Concatenate(axis=-1)(inputs[2] + [lstm1])
    print(k)
    # Dense layers
    dns1 = Dense(128, activation='relu')(lstm2)
    con3 = Dropout(0.1)(dns1)
    dns2 = Dense(64, activation='relu')(con3)
    dp1 = Dropout(0.2)(dns2)
    dns3 = Dense(1, activation='sigmoid')(dp1)

    # Model
    model = Model(inputs[0] + [cont_input, lstm_input], dns3)
    print(model.summary(90))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  sample_weight_mode='temporal',
                  metrics=['binary_crossentropy'])
    return model
