# Green Card Prediction App

## Overview
This project is a web application that predicts the probability of receiving a green card based on user inputs. It utilizes machine learning models, specifically `XGBoost`, to analyze various factors and provides recommendations for improving the chances of approval.

## Features
- User-friendly web interface built with Flask and React.
- Machine learning model trained on historical green card data.
- GPT-o3-Mini API integration for explanation and improvement suggestions.
- Data preprocessing and feature engineering for accurate predictions.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/sanyachawla26/datathon-applicationform
   ```

2. Navigate to the repository folder:
   ```sh
   cd <your-repo-path>
   ```

3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Set up the `OPENAI_API_KEY` environment variable:
   ```sh
   export OPENAI_API_KEY="your-api-key-here"
   ```

5. Run the Flask application:
   ```sh
   python app.py
   ```

6. Open a web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

## Project Structure
```
/project-root
│── /data                  # Folder for input data (ONLY NEEDED FOR TRAINING NEW MODELS)
│── /models                # Trained ML models
│── /templates
│   ├── index.html         # Frontend web interface
│── app.py                 # Flask web server
│── model.py               # Machine learning model implementation
│── requirements.txt       # Project dependencies
│── README.md              # Documentation
```

## API Endpoints

### `POST /receive-data`
- Accepts JSON input with user information.
- Returns the probability of green card approval and GPT-generated advice.

#### Sample Request
```json
{
    "countryOfCitizenship": "India",
    "classOfAdmission": "H-1B",
    "yearfiled": "2023",
    "foreignWorkerEducation": "Master's",
    "foreignWorkerInfoMajor": "Computer Science",
    "foreignWorkerInstOfEd": "XYZ University",
    "foreignWorkerReqExperience": "Y",
    "foreignWorkerExpWithEmpl": "N",
    "foreignWorkerCurrEmployed": "Y",
    "jobTitle": "Software Engineer",
    "minimumEducation": "Bachelor's",
    "majorFieldOfStudy": "Engineering",
    "foreignLanguageRequired": "N"
}
```

#### Sample Response
```json
{
    "probability": 0.85,
    "gpt_response": "Your chances of receiving a green card are high. However, gaining additional relevant work experience or certifications may improve your application."
}
```

## Authors
- Mattias Blum
- Sanya Chawla
- James Jung
- Jana Ka 
- Jason Majoros