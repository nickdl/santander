import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import pickle
from keras.preprocessing.sequence import pad_sequences


store_dtype = np.int8

key = ['fecha_dato', 'ncodpers']
continuous = ['age', 'antiguedad', 'fecha_alta', 'renta']
categorical = ['ind_empleado', 'indrel_1mes', 'tiprel_1mes', 'segmento']
binary = ['ind_nuevo', 'indrel', 'indresi', 'indext', 'indfall', 'ind_actividad_cliente', 'sexo']
unused = ['tipodom', 'ult_fec_cli_1t', 'conyuemp', 'nomprov'] + ['pais_residencia', 'canal_entrada', 'cod_prov']
target = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1',
          'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',
          'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
          'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
          'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

suffixes = {
    'ind_empleado': ['A', 'B', 'F', 'N', 'S'],
    'indrel_1mes': [1.0, 2.0, 3.0, 4.0, 5.0],
    'tiprel_1mes': ['A', 'I', 'P', 'R', 'N'],
    'segmento': ['02 - PARTICULARES', '03 - UNIVERSITARIO', '01 - TOP'],
}
categorical_expanded = [x + '_' + str(y) for x in categorical for y in suffixes[x]]

usecols = key + continuous + categorical + binary + target
dtype = {'ncodpers': np.int64}
dtype.update({x: np.float32 for x in continuous})
dtype.update({x: store_dtype for x in categorical_expanded + binary + target})


def sample_dataset(n_users=50000):
    df = pd.read_csv('data/train_ver2.csv', usecols=['ncodpers'], nrows=700000)
    sample_users = pd.Series(df['ncodpers'].unique()).sample(n=n_users)
    chunks = pd.read_csv('data/train_ver2.csv', usecols=usecols, chunksize=100000)
    sampled_data = pd.concat([chunk[chunk['ncodpers'].isin(sample_users)] for chunk in chunks])
    return sampled_data


def fit_scaler():
    df = sample_dataset()
    df.dropna(subset=['fecha_alta', 'ind_empleado', 'sexo', 'indrel_1mes', 'tiprel_1mes', 'ind_nuevo',
                      'ind_nuevo', 'indrel', 'indresi', 'indext', 'indfall', 'ind_actividad_cliente'], inplace=True)
    df.dropna(subset=target, inplace=True)
    df['renta'].fillna(134254.3, inplace=True)
    df["fecha_alta"] = pd.to_datetime(df["fecha_alta"], format="%Y-%m-%d").apply(lambda x: time.mktime(x.timetuple()))
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(df[continuous])
    with open('scaler.txt', 'wb') as s_file:
        pickle.dump(scaler, s_file)


def clean(df):
    df.dropna(subset=['fecha_alta', 'ind_empleado', 'sexo', 'indrel_1mes', 'tiprel_1mes', 'ind_nuevo',
                      'ind_nuevo', 'indrel', 'indresi', 'indext', 'indfall', 'ind_actividad_cliente'], inplace=True)
    df.dropna(subset=target, inplace=True)

    df['renta'].fillna(134254.3, inplace=True)
    df['segmento'].fillna('02 - PARTICULARES', inplace=True)

    df["fecha_dato"] = pd.to_datetime(df["fecha_dato"], format="%Y-%m-%d")
    df["fecha_alta"] = pd.to_datetime(df["fecha_alta"], format="%Y-%m-%d").apply(lambda x: time.mktime(x.timetuple()))

    # scaling
    with open('scaler.txt', 'rb') as s_file:
        scaler = pickle.load(s_file)
    df[continuous] = scaler.transform(df[continuous])
    df[continuous] = df[continuous].astype(np.float32)

    # one-hot encoding
    df['indrel_1mes'].replace(to_replace=['P'], value=5, inplace=True)
    df['indrel_1mes'] = pd.to_numeric(df['indrel_1mes'])

    for column, val_list in suffixes.items():
        for val in val_list:
            df[column + '_' + str(val)] = (df[column] == val).astype(store_dtype)
    df[categorical_expanded] = df[categorical_expanded].replace(to_replace=0, value=-1)
    df.drop(categorical, axis=1, inplace=True)

    # binary
    df[binary] = df[binary].replace(to_replace=['S', 'N'], value=[1, -1])
    df[binary] = df[binary].replace(to_replace=['H', 'V'], value=[1, -1])
    df['indrel'] = df['indrel'].replace(to_replace=[99.0, 1.0], value=[1, -1])
    df[['ind_nuevo', 'ind_actividad_cliente']] = df[['ind_nuevo', 'ind_actividad_cliente']]\
        .replace(to_replace=[0.0], value=[-1])
    df[binary] = df[binary].astype(store_dtype)

    df[target] = df[target].astype(store_dtype)
    return df


def clean_data():
    chunks = pd.read_csv('data/train_ver2.csv', usecols=usecols, chunksize=50000)
    df = pd.concat([clean(chunk) for chunk in chunks])
    df.to_csv('data_clean.csv', index=False)


def pad():
    users = pd.read_csv('data/test_ver2.csv', usecols=['ncodpers']).sort_values(by=['ncodpers'])['ncodpers'].tolist()
    data = pd.read_csv('data/data_clean.csv', dtype=dtype)

    seq_list, label_list_train, label_list_val, users_predicted = [], [], [], []
    for user in users:
        print(user)
        user_data = data[data['ncodpers'] == user].sort_values(by=['fecha_dato'])
        if user_data.shape[0] >= 3:
            seq = user_data[continuous + categorical_expanded + binary + target]
            seq_list.append(seq)
            users_predicted.append(user)
            labels = user_data[target]
            label_list_train.append(labels.iloc[-2].values.astype(np.float32))
            label_list_val.append(labels.iloc[-1].values.astype(np.float32))

    samples = pad_sequences(seq_list, maxlen=17, dtype=np.float32)
    np.save('data/data', samples)
    users_np = np.array(users_predicted)
    np.save('data/users', users_np)
    labels_train = np.array(label_list_train)
    np.save('data/labels_train', labels_train)
    labels_val = np.array(label_list_val)
    np.save('data/labels_val_all', labels_val)


def train_dataset():
    data = np.load('data/data.npy')
    data = np.concatenate((np.zeros((data.shape[0], 2, data.shape[2]), dtype=np.float32), data), axis=1)
    np.save('data/data_train', data[:, :17, :])


def val_dataset(size=200000):
    data = np.load('data/data.npy')
    labels = np.load('data/labels_val_all.npy')
    indices = np.random.choice(data.shape[0], size=size, replace=False)
    data = data[indices]
    labels = labels[indices]
    data = np.concatenate((np.zeros((data.shape[0], 1, data.shape[2]), dtype=np.float32), data), axis=1)
    np.save('data/data_val', data[:, :17, :])
    np.save('data/labels_val', labels)
