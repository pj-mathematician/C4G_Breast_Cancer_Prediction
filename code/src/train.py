import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sys import argv
from tqdm import tqdm

# Read data arg
data = pd.read_csv(argv[1])
data = data.iloc[:,1:].drop_duplicates()
data = data[['menopaus', 'agegrp', 'density', 'race', 'Hispanic', 'bmi', 'agefirst',
       'nrelbc', 'brstproc', 'lastmamm', 'surgmeno', 'hrt', 'invasive', 'count',
       'cancer']]
scaler = StandardScaler()
data.iloc[:,:-1] = scaler.fit_transform(data.iloc[:,:-1])
kf = KFold(n_splits = 5)
scores = []
models = []
models_x = []
for tr, te in tqdm(kf.split(data),  total = 5):
  X_train = data.iloc[tr,:-1]
  y_train = data.iloc[tr,-1]
  X_test = data.iloc[te,:-1]
  y_test = data.iloc[te,-1]
  kitty = CatBoostClassifier(random_state = 69)
  kitty_x = XGBClassifier()
  kitty.fit(X_train, y_train, verbose = False)
  kitty_x.fit(X_train, y_train, verbose = False)
  models.append(kitty)
  models_x.append(kitty_x)
print('Training done, saving models...')
model_path = argv[2]
# save catboost models
for i, model in enumerate(models):
    model.save_model(f'{model_path}/catboost_{i}.cbm')

# save xgboost models
for i, model in enumerate(models_x):
    model.save_model(f'{model_path}/xgboost_{i}.cbm')

print('Models Saved, code finished')