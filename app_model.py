from flask import Flask, request, jsonify
import pandas as pd
import os
import pickle
import sqlite3
from sklearn.linear_model import LinearRegression

os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/", methods=['GET'])
def hello():
    return "Bienvenido a mi API del modelo advertising"

# 0. Endpoint que devuelva la predicción de nuevos datos enviados mediante argumentos en la llamada
@app.route('/v1/predict', methods=['GET'])
def predict():
    model = pickle.load(open('data/advertising_model','rb'))

    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)

    if tv is None or radio is None or newspaper is None:
        return "Missing args, the input values are needed to predict"
    else:
        prediction = model.predict([[int(tv),int(radio),int(newspaper)]])
        return "The prediction of sales investing that amount of money in TV, radio and newspaper is: " + str(round(prediction[0],2)) + 'k €'
    

# 1. Endpoint que ofrezca la predicción de ventas a partir de todos los valores de gastos en publicidad. (/v2/predict)
@app.route('/v2/predict_bd', methods=['GET'])

def predict_bd():
    query = "SELECT * FROM Advertising;"
    conn = sqlite3.connect('.\\ejercicio\\data\\Advertising.db')
    crsr = conn.cursor()
    crsr.execute(query)
    ans = crsr.fetchall()
    conn.close()
    names = [description[0] for description in crsr.description]
    df=pd.DataFrame(ans,columns=names).drop(columns=['sales'])
    model = pickle.load(open('data/advertising_model','rb'))

    predictions = model.predict(df)

    rounded_predictions = [round(pred, 2) for pred in predictions]

    
    predictions_dict = {"predictions": rounded_predictions}

    return jsonify(predictions_dict)
    


#2. Un endpoint para almacenar nuevos registros en la base de datos que deberás crear previamente.(/v2/ingest_data)



@app.route('/v2/ingest_data', methods=['POST'])
def add_register():
    
    # Extract data from the request's JSON payload
    data = request.get_json()
    query = "INSERT INTO Advertising (tv, radio, newspaper,sales) VALUES (?, ?, ?,?)"
    tv = data.get('tv')
    radio = data.get('radio')
    newspaper = data.get('newspaper')
    sales = data.get('sales')

    connection = sqlite3.connect('.\\ejercicio\\data\\Advertising.db')
    crsr = connection.cursor()
    crsr.execute(query,(tv, radio, newspaper,sales))
    connection.commit()
    connection.close()

    return jsonify({'message': 'New book record added successfully'})

#3. Posibilidad de reentrenar de nuevo el modelo con los posibles nuevos registros que se recojan. (/v2/retrain)
@app.route('/v2/retrain', methods=['POST'])
def retrain():
    query = "SELECT * FROM Advertising;"
    conn = sqlite3.connect('.\\ejercicio\\data\\Advertising.db')
    crsr = conn.cursor()
    crsr.execute(query)
    ans = crsr.fetchall()
    conn.close()
    names = [description[0] for description in crsr.description]
    df=pd.DataFrame(ans,columns=names)
    X=df.drop(columns=['sales'])
    y=df['sales']
    model= LinearRegression()
    model.fit(X,y)
    filename = 'new_model'
    pickle.dump(model, open(filename, 'wb'))
    return jsonify({'message': 'New model trained'})

app.run()