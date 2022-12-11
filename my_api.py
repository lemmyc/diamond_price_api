from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler



from scipy import stats

import pandas as pd
import numpy as np

# Khởi tạo Flask Server Backend
app = Flask(__name__)



# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = "static"




data_df = pd.read_csv("./Diamonds_Prices2022.csv", index_col = 0)

si = SimpleImputer(missing_values = 0.00, strategy = 'mean')

data_df.iloc[:, 7:10] = si.fit_transform(data_df.iloc[:, 7:10])


num_cols = data_df.select_dtypes('number')
data_df = data_df[(np.abs(stats.zscore(num_cols)) < 3).all(axis=1)]
data_df = data_df.reset_index().iloc[:, 1:]
X = data_df.iloc[:, [0,1,2,3,4,5,7,8,9]]
Y = data_df.iloc[:, 6]

depth_mean = data_df['depth'].mean()
depth_std = data_df['depth'].std()

table_mean = data_df['table'].mean()
table_std = data_df['table'].std()

le = LabelEncoder()
for i in range(1, 4):
  X.iloc[:, i] = le.fit_transform(X.iloc[:, i])
sc = StandardScaler()

X.iloc[:, 4:6] = sc.fit_transform(X.iloc[:, 4:6])


X = X.values
y = Y.values
dtr_model = DecisionTreeRegressor(random_state = 10).fit(X, y)

@app.route('/', methods=['POST'] )
def predict_dtr():
    data = request.get_json()

    
    if data:
        input_x  = []
        for key, value in data.items():
            if(key == 'depth'):
                input_x.append((float(value) - depth_mean)/depth_std)
            elif (key == 'table'):
                input_x.append((float(value) - table_mean)/table_std)
            else:
                input_x.append(float(value))
        npa_input = np.array([input_x])
        price = dtr_model.predict(npa_input)

        return str(price[0])

    return 'API response'

# Start Backend
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='6868')