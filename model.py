from keras.models import Sequential, load_model
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense
from keras.callbacks import EarlyStopping
from process_data import *
from average_precision import mapk


input_size = 53
input_length = 17
output_size = 24
batch_size = 256


def get_model():
    model = Sequential()
    model.add(LSTM(100, batch_input_shape=(batch_size, input_length, input_size), return_sequences=False,
                   dropout_W=0.1, dropout_U=0.1))
    model.add(Dense(output_size, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train(model_file):
    model = get_model()

    data_train = np.load('data/data_train.npy')
    labels_train = np.load('data/labels_train.npy')
    data_val = np.load('data/data_val.npy')
    labels_val = np.load('data/labels_val.npy')

    early_stopping = EarlyStopping(monitor='val_loss', patience=0, mode='auto')

    model.fit(data_train, labels_train, batch_size=batch_size, nb_epoch=1, verbose=2, callbacks=[early_stopping],
              validation_split=0.0, validation_data=(data_val, labels_val), shuffle=True)

    model.save('models/' + model_file)
    score, acc = model.evaluate(data_val, labels_val, batch_size=batch_size)

    print('Val score:', score)
    print('Val accuracy:', acc)


def predict(model_file, out_file_path):
    model = load_model('models/' + model_file)
    data = np.load('data/data.npy')
    users_predicted = np.load('data/users.npy')
    users = pd.read_csv('data/test_ver2.csv', usecols=['ncodpers'])\
        .sort_values(by=['ncodpers'])['ncodpers'].values.tolist()

    prediction = model.predict(data, batch_size=batch_size)

    with open('results/' + out_file_path, 'w') as out_file:
        out_file.write('ncodpers,added_products\n')
        for user in users:
            index = np.where(users_predicted == user)[0]
            if index:
                user_prediction = prediction[index]
                recommended = []
                zipped_list = sorted(zip(target, user_prediction[0], data[index, -1, -output_size:][0]),
                                     key=lambda x: x[1], reverse=True)
                for label, p_prediction, last_month in zipped_list:
                    if not last_month:
                        recommended.append(label)
            else:
                recommended = ['ind_cco_fin_ult1', 'ind_ctop_fin_ult1', 'ind_recibo_ult1', 'ind_ecue_fin_ult1',
                               'ind_cno_fin_ult1', 'ind_nom_pens_ult1', 'ind_ctpp_fin_ult1']
            out_file.write(str(user) + ',' + ' '.join(recommended[:7]) + '\n')


def map7(model_file):
    model = load_model('models/' + model_file)
    data_val = np.load('data/data_val.npy')
    labels_val = np.load('data/labels_val.npy')

    prediction = model.predict(data_val, batch_size=batch_size)
    recommended_list = []
    for index in range(data_val.shape[0]):
        user_prediction = prediction[index]
        recommended = []
        zipped_list = sorted(zip(list(range(output_size)), user_prediction, data_val[index, -1, -output_size:]),
                             key=lambda x: x[1], reverse=True)
        for label_index, p_prediction, last_month in zipped_list:
            if not last_month:
                recommended.append(label_index)
        recommended_list.append(recommended[:7])

    added_products = np.where(labels_val - data_val[:, -1, -output_size:] == 1.0)
    added_products_list = [[] for _ in range(data_val.shape[0])]
    for i in range(added_products[0].shape[0]):
        added_products_list[added_products[0][i]].append(added_products[1][i])

    print(mapk(added_products_list, recommended_list, k=7))


# train('v1.h5')
# map7('v1.h5')
# predict('v1.h5', 'r1.csv')
