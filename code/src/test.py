import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sys import argv
from tqdm import tqdm

# Read training data arg
data = pd.read_csv(argv[1])
data = data.iloc[:,1:].drop_duplicates()
data = data[['menopaus', 'agegrp', 'density', 'race', 'Hispanic', 'bmi', 'agefirst',
       'nrelbc', 'brstproc', 'lastmamm', 'surgmeno', 'hrt', 'invasive', 'count',
       'cancer']]
scaler = StandardScaler()
data.iloc[:,:-1] = scaler.fit_transform(data.iloc[:,:-1])

# read test data arg
test = pd.read_csv(argv[2])

# scale test data
testids = test.iloc[:,0:1]
test.iloc[:,1:] = scaler.transform(test.iloc[:,1:])

# read models by arg
model_path = argv[3]
models = []
models_x = []
for i in range(5):
    models.append(CatBoostClassifier())
    models[i].load_model(f'{model_path}/catboost_{i}.cbm')
    models_x.append(XGBClassifier())
    models_x[i].load_model(f'{model_path}/xgboost_{i}.cbm')

for i, kitty in enumerate(models):
  testids['pred_{}'.format(i)] = kitty.predict_proba(test.iloc[:,1:])[:,1]
  testids['pred_{}_x'.format(i)] = models_x[i].predict_proba(test.iloc[:,1:])[:,1]

testids['prediction'] = (sum([testids['pred_{}'.format(i)] for i in range(5)])*10 + sum([testids['pred_{}_x'.format(i)] for i in range(5)])*1)/(55)
solution_path = argv[4]
testids[['id','prediction']].to_csv(solution_path,index = False)

print('testing_done, solution saved at {}'.format(solution_path))