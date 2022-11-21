#import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import pandas as pd
import mlflow
import os
#from mlflow.tracking import MlflowClient

os.environ["AWS_PROFILE"] = "meet9426"
#------------------------------------------------------------------------# 
RUN_ID = "04bbfaf05448493ab222a1a24e963b4b"
logged_model = f"s3://mlflow-artifacts-rmte/2/{RUN_ID}/artifacts/model"
loaded_model = mlflow.pyfunc.load_model(logged_model)
#------------------------------------------------------------------------#
    
app = Flask("claims")

def prepare_feat(df):
    df['GENDER'] = df.GENDER.replace({'female':0,'male':1})
    df['DRIVING_EXPERIENCE'] = df.DRIVING_EXPERIENCE.replace({'0-9y':0,'10-19y':1,'20-29y':2,'30y+':3})
    df['TYPE_OF_VEHICLE'] = df.TYPE_OF_VEHICLE.replace({'HatchBack':0,'Sedan':1,'SUV':2,'Sports Car':3})
    
    df.loc[df.SPEEDING_VIOLATIONS>=3,'SPEEDING_VIOLATIONS'] = 3
    df.loc[df.PAST_ACCIDENTS>=4,'PAST_ACCIDENTS'] = 4
    df.loc[df.DUIS>=3,'DUIS'] = 3
    
    return df


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    print(request.form.get('GENDER'))
    print(request.form.get('DRIVING_EXPERIENCE'))
    print(type(request.form.get('TYPE_OF_VEHICLE')))
    print(request.form.get('SPEEDING_VIOLATIONS'))
    print(type(request.form.get('SPEEDING_VIOLATIONS')))
    #print(request.form.get('GENDER'))
    data_dict = {'GENDER':request.form.get('GENDER'),
                 'DRIVING_EXPERIENCE':request.form.get('DRIVING_EXPERIENCE'),
                 'TYPE_OF_VEHICLE':request.form.get('TYPE_OF_VEHICLE'),
                 'SPEEDING_VIOLATIONS':float(request.form.get('SPEEDING_VIOLATIONS')),
                 'PAST_ACCIDENTS':float(request.form.get('PAST_ACCIDENTS')),
                 'DUIS':float(request.form.get('DUIS'))}  
    data = pd.DataFrame(data_dict,index=range(1))
    pred = loaded_model.predict(prepare_feat(data))

    print("#------------------------------------------------------------------------#",'\n')
    print("#--------------------------------Result is-------------------------------#",'\n')
    print(pred)

    result = {
                'Claim pass or Fail' : pred[0]
             }
    
    return render_template("home.html",prediction_text="The claim prediction is {}".format(pred[0]))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data_dict = {'GENDER':request.form.get('GENDER'),
                 'DRIVING_EXPERIENCE':request.form.get('DRIVING_EXPERIENCE'),
                 'TYPE_OF_VEHICLE':request.form.get('TYPE_OF_VEHICLE'),
                 'SPEEDING_VIOLATIONS':float(request.form.get('SPEEDING_VIOLATIONS')),
                 'PAST_ACCIDENTS':float(request.form.get('PAST_ACCIDENTS')),
                 'DUIS':float(request.form.get('DUIS'))}
    data = pd.DataFrame(data_dict,index=range(1))
    pred = loaded_model.predict(prepare_feat(data))

    print("#------------------------------------------------------------------------#",'\n')
    print("#--------------------------------Result is-------------------------------#",'\n')
    print(pred)

    result = {
                'Claim pass or Fail' : pred[0]
             }

    return jsonify(result)
##############################

@app.route('/predict_api_json', methods=['POST'])
def predict_api_json():
    ip = request.get_json()
    print(ip)
    data = pd.DataFrame(ip,index=range(1))
    pred = loaded_model.predict(prepare_feat(data))

    print("#------------------------------------------------------------------------#",'\n')
    print("#--------------------------------Result is-------------------------------#",'\n')
    print(pred)

    result = {
                'Claim pass or Fail' : pred[0]
             }

    return jsonify(result)

##############################


if __name__ == "__main__":
    app.run(debug=True)
    # , host='0.0.0.0', port=9696