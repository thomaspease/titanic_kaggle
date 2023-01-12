import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder


TRAIN_PATH = '/Users/tompease/Documents/Coding/titanic/data/cleaned_train.csv'
TEST_PATH = '/Users/tompease/Documents/Coding/titanic/data/cleaned_test.csv'

def encode(row):
  encoder = OrdinalEncoder()
  output = encoder.fit_transform(row)
  return output

# These two functions are suboptimal
def encode_test(train_column, test_column):
  encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
  encoder.fit(train_column)
  output = encoder.transform(test_column)
  return output

def scale_test(train, test):
  scaler = MinMaxScaler()
  scaler.fit(train)
  output = scaler.transform(test)
  return output


class TitanicLoader():
  def __init__(self):
    self.complete_train = pd.read_csv(TRAIN_PATH)
    # self.numeric_df = self.complete_train.select_dtypes(['number'])
    self.complete_test = pd.read_csv(TEST_PATH)

  def load(self, label):
    df = self.complete_train.copy()
    
    enc_columns = ['Sex', 'Cabin', 'Embarked']
    df.loc[:, enc_columns] = encode(df.loc[:, enc_columns])

    df = df.drop(['PassengerId', 'Unnamed: 0'], axis=1)

    # Normalize fare and age
    scaler = MinMaxScaler()
    df.loc[:, ['Age', 'Fare']] = scaler.fit_transform(df.loc[:, ['Age', 'Fare']])

    features_df = df.drop([label], axis=1)
    label_df = df.loc[:, label]

    return features_df, label_df

  def load_test(self):
    df = self.complete_test.copy()
    
    # fandango to try to get them encoded in the same way as the train data
    enc_columns = ['Sex', 'Cabin', 'Embarked']
    df.loc[:, enc_columns] = encode_test(self.complete_train.loc[:, enc_columns], df.loc[:, enc_columns])

    df = df.drop(['PassengerId', 'Unnamed: 0'], axis=1)

    # Normalize fare and age
    scaler = MinMaxScaler()
    df.loc[:, ['Age', 'Fare']] = scale_test(self.complete_train.loc[:, ['Age', 'Fare']], df.loc[:, ['Age', 'Fare']])

    return df

loader = TitanicLoader()
x, y = loader.load('Survived')
test = loader.load_test()

