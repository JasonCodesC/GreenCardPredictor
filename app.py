import os
import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify, render_template
from model import GreenCardPredictor
from openai import OpenAI
import numpy as np

key = "key"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", key))

app = Flask(__name__)

# Initialize predictor and load/train model
MODEL_PATH = "models/xgboost/model.json"
predictor = GreenCardPredictor(root_path="./", input_path="data", models_path="models")

#if os.path.exists(MODEL_PATH):
#    predictor.model = xgb.XGBClassifier()
#    predictor.model.load_model(MODEL_PATH)
#else:
predictor.load_data()
predictor.filter_data()
predictor.preprocess_data()
predictor.train_model()
    #predictor.model.save_model(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/receive-data', methods=['POST'])
def receive_data():
    data = request.get_json()
    model_column_mappings = {
        "countryOfCitizenship" : "COUNTRY_OF_CITIZENSHIP",
        "classOfAdmission" : "CLASS_OF_ADMISSION",
        "yearfiled" : "YEAR_FILED",
        "foreignWorkerEducation" : "FOREIGN_WORKER_EDUCATION",
        "foreignWorkerInfoMajor" : "FOREIGN_WORKER_INFO_MAJOR",
        "foreignWorkerInstOfEd" : "FOREIGN_WORKER_INST_OF_ED",
        "foreignWorkerReqExperience" : "FOREIGN_WORKER_REQ_EXPERIENCE",
        "foreignWorkerExpWithEmpl" : "FOREIGN_WORKER_EXP_WITH_EMPL",
        "foreignWorkerCurrEmployed" : "FOREIGN_WORKER_CURR_EMPLOYED",
        "jobTitle" : "JOB_TITLE",
        "minimumEducation" : "MINIMUM_EDUCATION",
        "majorFieldOfStudy" : "MAJOR_FIELD_OF_STUDY",
        "foreignLanguageRequired" : "FOREIGN_LANGUAGE_REQUIRED",
    }
    data = {model_column_mappings.get(k, k): (np.nan if v in ['', 'select'] else v) for k, v in data.items()}
    print("Received JSON Data:", data)  # Debugging print statement
    user_df = pd.DataFrame(data, index=[0]).reset_index(drop=True)
    user_df['CASE_NUMBER'] = 'USER'
    filtered_user_df = predictor.filter_data(user_df)
    preprocessed_user_df = predictor.preprocess_data(filtered_user_df)

    proba = predictor.model.predict_proba(preprocessed_user_df.drop(columns=[predictor.id_column]))[:, 1][0]
    important_missing_features = predictor.identify_important_missing_features(user_df)
    proba = float(proba)

    gpt_prompt = f"""
        The model predicts a {round(proba,3) * 100}% probability of receiving a green card.
        The most important missing features affecting this prediction according to their aggregated SNAP values are: {important_missing_features}.
        Provide a text summary of these results (in layman's terms) as well as advice on how to improve their chances based on which features are currently missing.
        """
    #print("GPT Prompt: ", gpt_prompt)

    response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[{"role": "developer", "content": "You are a helpful assistant that summarizes our output"},
    {"role": "user", "content": gpt_prompt}],
    max_tokens=150 
    )

   # print(response.choices[0].message.content)
    gpt_message = response.choices[0].message.content.strip()
        
  #  print('gpt_msg' + gpt_message)
    return jsonify({"probability": proba, "gpt_response": gpt_message}), 200

if __name__ == '__main__':
    app.run(debug=True)


