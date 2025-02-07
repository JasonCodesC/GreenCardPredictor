import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, KeyedVectors
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import xgboost as xgb
import shap
pd.options.mode.chained_assignment = None  # default='warn'

class GreenCardPredictor():
    def __init__(self, root_path, input_path, models_path):
        self.root_path = root_path
        self.input_path = os.path.join(root_path, input_path)
        self.models_path = os.path.join(root_path, models_path)
        self.data = pd.DataFrame()
        self.preprocessed_data = None
        self.model = None
        self.id_column = 'CASE_NUMBER'
        self.target_column = 'CASE_STATUS'

        # Features an Applicant Would Likely Know
        self.personal_background_features = [
            "COUNTRY_OF_CITIZENSHIP",
            "CLASS_OF_ADMISSION",
            "YEAR_FILED"
        ]

        self.education_training_features = [
            "FOREIGN_WORKER_EDUCATION",
            "FOREIGN_WORKER_INFO_MAJOR",
            "FOREIGN_WORKER_INST_OF_ED"
        ]

        self.work_experience_features = [
            "FOREIGN_WORKER_REQ_EXPERIENCE",
            "FOREIGN_WORKER_EXP_WITH_EMPL",
            "FOREIGN_WORKER_CURR_EMPLOYED"
        ]

        self.job_employer_related_features = [
            "JOB_TITLE",
            "MINIMUM_EDUCATION",
            "MAJOR_FIELD_OF_STUDY",
            "FOREIGN_LANGUAGE_REQUIRED",
        ]

        self.columns_of_interest = [self.id_column] + \
                                   self.personal_background_features + \
                                   self.education_training_features + \
                                   self.work_experience_features + \
                                   self.job_employer_related_features + \
                                   [self.target_column]

        self.ohe_features = []

    def load_data(self):
        for filename in os.listdir(self.input_path):
            if any(item in filename for item in ['22','23','24']):
                year = None
                if '22' in filename:
                    year = '2022'
                elif '23' in filename:
                    year = '2023'
                elif '24' in filename:
                    year = '2024'
                path = os.path.join(self.input_path, filename)
                df = pd.read_csv(path, low_memory=False, encoding_errors='replace')
                df['YEAR_FILED'] = year
                print(path)
                self.data = pd.concat([self.data, df])
        self.data = self.data[self.columns_of_interest]
        print(f"Data Loaded. Shape: {self.data.shape}")
        return self.data

    def get_personal_background_data(self, data=None):
        if data is None:
            data = self.data
        return data[[self.id_column]+self.personal_background_features]

    def get_education_training_data(self, data=None):
        if data is None:
            data = self.data
        return data[[self.id_column]+self.education_training_features]

    def get_work_experience_data(self, data=None):
        if data is None:
            data = self.data
        return data[[self.id_column]+self.work_experience_features]

    def get_job_employer_related_data(self, data=None):
        if data is None:
            data = self.data
        return data[[self.id_column]+self.job_employer_related_features]

    def filter_personal_background_data(self, data=None):
        modify_self = data is None
        data = self.data if modify_self else data.copy()

        data['CLASS_OF_ADMISSION'] = data['CLASS_OF_ADMISSION'].fillna('No Visa/Not in USA')
        data['CLASS_OF_ADMISSION'] = data['CLASS_OF_ADMISSION'].replace('Not in USA', 'No Visa/Not in USA')
        data.dropna(subset=self.personal_background_features, inplace=True)
        data.reset_index(drop=True, inplace=True)

        if modify_self:
            self.data = data
        return data

    def filter_education_training_data(self, data=None):
        modify_self = data is None
        data = self.data if modify_self else data.copy()

        data['FOREIGN_WORKER_EDUCATION'] = data['FOREIGN_WORKER_EDUCATION'].fillna('None')
        data['FOREIGN_WORKER_INST_OF_ED'] = data['FOREIGN_WORKER_INST_OF_ED'].fillna('N/A')
        data['FOREIGN_WORKER_INFO_MAJOR'] = data['FOREIGN_WORKER_INFO_MAJOR'].fillna('N/A')
        data.dropna(subset=self.education_training_features, inplace=True)
        data.reset_index(drop=True, inplace=True)

        if modify_self:
            self.data = data
        return data

    def filter_work_experience_data(self, data=None):
        modify_self = data is None
        data = self.data if modify_self else data.copy()

        data['FOREIGN_WORKER_REQ_EXPERIENCE'] = data['FOREIGN_WORKER_REQ_EXPERIENCE'].fillna('N/A')
        data['FOREIGN_WORKER_EXP_WITH_EMPL'] = data['FOREIGN_WORKER_EXP_WITH_EMPL'].fillna('N/A')
        data.dropna(subset=self.work_experience_features, inplace=True)
        data.reset_index(drop=True, inplace=True)

        if modify_self:
            self.data = data
        return data

    def filter_job_employer_related_data(self, data=None):
        modify_self = data is None
        data = self.data if modify_self else data.copy()
        data['MINIMUM_EDUCATION'] = data['MINIMUM_EDUCATION'].fillna('Not Required')
        data['MAJOR_FIELD_OF_STUDY'] = data['MAJOR_FIELD_OF_STUDY'].fillna('N/A')
        data.dropna(subset=self.job_employer_related_features, inplace=True)
        data.reset_index(drop=True, inplace=True)

        if modify_self:
            self.data = data
        return data

    def filter_target_columns(self):
        self.data = self.data[self.data[self.target_column] != 'Withdrawn']
        self.data[self.target_column] = self.data[self.target_column].replace('Certified-Expired', 'Certified')
        self.data.reset_index(drop=True, inplace=True)

    def filter_data(self, data=None):
        modify_self = data is None
        if modify_self:
            data = self.data
        else:
            data = data.copy()

        data = self.filter_personal_background_data(data)
        data = self.filter_education_training_data(data)
        data = self.filter_work_experience_data(data)
        data = self.filter_job_employer_related_data(data)

        data.drop_duplicates(inplace=True)
        data.reset_index(drop=True, inplace=True)

        if modify_self:
            self.data = data
            self.filter_target_columns()

        print(f"Filtered new dataset. Shape: {data.shape}")
        return data

    def train_word2vec_model(self, text_column, vector_size=100):
        text_data = self.data[text_column].values.flatten()

        # Tokenize text
        sentences = [str(text).lower().split() for text in text_data]

        # Train Word2Vec model
        word2vec_model = Word2Vec(sentences, vector_size=vector_size, window=5, min_count=1, workers=4)

        # Save the trained model
        path_to_save = os.path.join(self.models_path, 'word2vec', f'{text_column}.model')
        word2vec_model.save(path_to_save)
        return word2vec_model

    def load_word2vec_model(self, text_column):
        # Load saved model for that location
        path_to_load = os.path.join(self.models_path, 'word2vec', f'{text_column}.model')

        # Try to load exisiting model, if it doesn't exist, train it
        try:
            word2vec_model = Word2Vec.load(path_to_load)
        except:
            word2vec_model = self.train_word2vec_model(text_column)
        finally:
            return word2vec_model

    def get_word2vec_embedding(self, text, model, vector_size=100):
        # Tokenize text
        words = str(text).lower().split()

        # Get word embeddings
        vectors = [model.wv[word] for word in words if word in model.wv]

        # Average word vectors
        if len(vectors) > 0:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(vector_size)

    def preprocess_personal_background_data(self, data=None):
        modify_self = data is None
        data = self.data if modify_self else data.copy()

        if not hasattr(self, 'encoders'):
            self.encoders = {}

        encoded_dfs = []

        # One-Hot encode categorical features
        for col in ['COUNTRY_OF_CITIZENSHIP', 'CLASS_OF_ADMISSION', 'YEAR_FILED']:
            if col not in self.encoders:
                self.encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded = self.encoders[col].fit_transform(data[[col]])
            else:
                encoded = self.encoders[col].transform(data[[col]])
            self.ohe_features.append(col)
            encoded_dfs.append(pd.DataFrame(encoded, columns=self.encoders[col].get_feature_names_out()))

        preprocessed_df = pd.concat([data[['CASE_NUMBER']]] + encoded_dfs, axis=1)

        if modify_self:
            self.preprocessed_data = preprocessed_df
        return preprocessed_df

    def preprocess_education_data(self, data=None):
        modify_self = data is None
        data = self.data if modify_self else data.copy()

        if not hasattr(self, 'encoders'):
            self.encoders = {}

        # One-Hot encode categorical features
        col = 'FOREIGN_WORKER_EDUCATION'
        if col not in self.encoders:
            self.encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = self.encoders[col].fit_transform(data[[col]])
        else:
            encoded = self.encoders[col].transform(data[[col]])
        self.ohe_features.append(col)
        edu_encoded_df = pd.DataFrame(encoded, columns=self.encoders[col].get_feature_names_out())

        # Get word2vec embeddings
        major_model = self.load_word2vec_model('FOREIGN_WORKER_INFO_MAJOR')
        inst_model = self.load_word2vec_model('FOREIGN_WORKER_INST_OF_ED')

        major_embeddings = np.array([self.get_word2vec_embedding(text, major_model) for text in data['FOREIGN_WORKER_INFO_MAJOR']])
        inst_embeddings = np.array([self.get_word2vec_embedding(text, inst_model) for text in data['FOREIGN_WORKER_INST_OF_ED']])

        major_embeddings_df = pd.DataFrame(major_embeddings, columns=[f'FOREIGN_WORKER_INFO_MAJOR_EMBEDDING_{i}' for i in range(major_embeddings.shape[1])])
        inst_embeddings_df = pd.DataFrame(inst_embeddings, columns=[f'FOREIGN_WORKER_INST_OF_ED_EMBEDDING_{i}' for i in range(inst_embeddings.shape[1])])

        preprocessed_df = pd.concat([data[['CASE_NUMBER']], edu_encoded_df, major_embeddings_df, inst_embeddings_df], axis=1)

        if modify_self:
            self.preprocessed_data = preprocessed_df
        return preprocessed_df

    def preprocess_work_experience_data(self, data=None):
        modify_self = data is None
        data = self.data if modify_self else data.copy()

        if not hasattr(self, 'encoders'):
            self.encoders = {}

        encoded_dfs = []

        for col in ['FOREIGN_WORKER_REQ_EXPERIENCE', 'FOREIGN_WORKER_EXP_WITH_EMPL', 'FOREIGN_WORKER_CURR_EMPLOYED']:
            if col not in self.encoders:
                self.encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded = self.encoders[col].fit_transform(data[[col]])
            else:
                encoded = self.encoders[col].transform(data[[col]])
            self.ohe_features.append(col)
            encoded_dfs.append(pd.DataFrame(encoded, columns=self.encoders[col].get_feature_names_out()))

        preprocessed_df = pd.concat([data[['CASE_NUMBER']]] + encoded_dfs, axis=1)

        if modify_self:
            self.preprocessed_data = preprocessed_df
        return preprocessed_df

    def preprocess_job_employer_related_data(self, data=None):
        modify_self = data is None
        data = self.data if modify_self else data.copy()

        if not hasattr(self, 'encoders'):
            self.encoders = {}

        encoded_dfs = []

        for col in ['MINIMUM_EDUCATION', 'FOREIGN_LANGUAGE_REQUIRED']:
            if col not in self.encoders:
                self.encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded = self.encoders[col].fit_transform(data[[col]])
            else:
                encoded = self.encoders[col].transform(data[[col]])
            self.ohe_features.append(col)
            encoded_dfs.append(pd.DataFrame(encoded, columns=self.encoders[col].get_feature_names_out()))

        major_model = self.load_word2vec_model('MAJOR_FIELD_OF_STUDY')
        job_title_model = self.load_word2vec_model('JOB_TITLE')

        major_embeddings = np.array([self.get_word2vec_embedding(text, major_model) for text in data['MAJOR_FIELD_OF_STUDY']])
        job_title_embeddings = np.array([self.get_word2vec_embedding(text, job_title_model) for text in data['JOB_TITLE']])

        major_embeddings_df = pd.DataFrame(major_embeddings, columns=[f'MAJOR_FIELD_OF_STUDY_EMBEDDING_{i}' for i in range(major_embeddings.shape[1])])
        job_title_embeddings_df = pd.DataFrame(job_title_embeddings, columns=[f'JOB_TITLE_EMBEDDING_{i}' for i in range(job_title_embeddings.shape[1])])

        preprocessed_df = pd.concat([data[['CASE_NUMBER']]] + encoded_dfs + [major_embeddings_df, job_title_embeddings_df], axis=1)

        if modify_self:
            self.preprocessed_data = preprocessed_df
        return preprocessed_df

    def preprocess_data(self, data=None):
        modify_self = data is None
        data = self.data if modify_self else data.copy()

        preprocessed_data = None
        preprocessing_functions = [
            self.preprocess_personal_background_data,
            self.preprocess_education_data,
            self.preprocess_work_experience_data,
            self.preprocess_job_employer_related_data
        ]

        for f in preprocessing_functions:
            if preprocessed_data is None:
                preprocessed_data = f(data)
            else:
                preprocessed_data = pd.merge(preprocessed_data, f(data), on='CASE_NUMBER')

        preprocessed_data.drop_duplicates(inplace=True)
        preprocessed_data.reset_index(drop=True, inplace=True)

        if modify_self:
            preprocessed_data[self.target_column] = data[self.target_column].map({'Certified': 1, 'Denied': 0})
            self.preprocessed_data = preprocessed_data

        print(f"Data Preprocessed. Shape: {preprocessed_data.shape}")
        return preprocessed_data

    def train_model(self, model=xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss'), show_metrics=False, param_grid=None, scoring='accuracy', cv=5, test_size=0.2, random_state=42):
        X = self.preprocessed_data.drop(columns=[self.target_column, self.id_column])
        y = self.preprocessed_data[self.target_column]

        # Compute class weights to handle imbalance
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

        # Split into train, val, test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

        if param_grid:
            grid_search = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train, sample_weight=[class_weight_dict[label] for label in y_train])
            print(f"Best parameters found: {grid_search.best_params_}")
            self.model = grid_search.best_estimator_
        else:
            model.fit(X_train, y_train, sample_weight=[class_weight_dict[label] for label in y_train])
            self.model = model

        if show_metrics:
            y_pred = self.model.predict(X_test)

            # Dummy Classifier Accuracy
            dummy_accuracy = accuracy_score(y_test, np.ones(len(y_test)))
            print(f"Dummy Classifier Accuracy: {dummy_accuracy}")

            # Show accuracy
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Test Accuracy: {accuracy}")

            # Compute and display confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.show()

            # ROC curve and AUC score
            y_test_prob = self.model.predict_proba(X_test)[:, 1]  # Assuming binary classification
            fpr, tpr, _ = roc_curve(y_test, y_test_prob)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
            plt.show()

    def identify_important_missing_features(self, raw_df):
        missing_raw_features = raw_df.columns[raw_df.isna().any()].tolist()

        if not missing_raw_features:
            print("No missing raw features detected.")
            return {}

        filtered_df = self.filter_data(raw_df)
        preprocessed_df = self.preprocess_data(filtered_df)

        feature_mapping = self.get_feature_mapping(preprocessed_df)

        related_processed_features = []
        for raw_feature in missing_raw_features:
            related_features = [processed for processed, raw in feature_mapping.items() if raw == raw_feature]
            related_processed_features.extend(related_features)

        if not related_processed_features:
            print("No corresponding processed features found for missing raw features.")
            return {}

        explainer = shap.Explainer(self.model)
        shap_values = explainer(preprocessed_df.drop([self.id_column], axis=1))

        # Compute mean absolute SHAP values for related processed features
        processed_feature_importance = {}

        for feat in related_processed_features:
            if feat in preprocessed_df.columns:  # Ensure feature exists in preprocessed data
                feat_index = np.where(preprocessed_df.columns == feat)[0]
                if feat_index.size > 0:
                    processed_feature_importance[feat] = np.abs(shap_values.values[:, feat_index[0]]).mean()

        # Aggregate importance by raw feature
        raw_feature_importance = {}

        for processed_feat, importance in processed_feature_importance.items():
            raw_feature = feature_mapping.get(processed_feat, processed_feat)  # Map back to raw feature
            raw_feature_importance[raw_feature] = raw_feature_importance.get(raw_feature, 0) + importance

        # Rank missing raw features by total SHAP impact
        ranked_missing_features = sorted(raw_feature_importance.items(), key=lambda x: x[1], reverse=True)

        return ranked_missing_features

    def get_feature_mapping(self, preprocessed_df):
        feature_mapping = {}

        # Collect all raw feature names
        raw_features = self.personal_background_features + \
                       self.education_training_features + \
                       self.work_experience_features + \
                       self.job_employer_related_features

        # Loop through raw feature names and find corresponding processed features
        for raw_feature in raw_features:
            for processed_feature in preprocessed_df.columns:
                if raw_feature in processed_feature:  # Processed feature contains raw feature name
                    feature_mapping[processed_feature] = raw_feature

        return feature_mapping


    def plot_historgrams(self):
        # Define feature categories
        feature_categories = {
            "Target Variable": ["CASE_STATUS"],
            "Personal Background": self.personal_background_features,
            "Education & Training": self.education_training_features,
            "Work Experience": self.work_experience_features,
            "Job & Employer Related": self.job_employer_related_features
        }

        # Create histograms
        for category, features in feature_categories.items():
            print(f"\nPlotting histograms for: {category}")

            for feature in features:
                if feature in self.data.columns:
                    plt.figure(figsize=(12, 5))

                    self.data[feature].value_counts().nlargest(100).plot(kind='bar', color='skyblue', edgecolor='black')
                    plt.xlabel(feature)
                    plt.ylabel("Count")
                    plt.title(f"Bar Chart of {feature}")
                    plt.xticks(rotation=45)

                    plt.grid(axis="y", linestyle="--", alpha=0.7)
                    plt.show()

    def compute_cosine_similarity(self, prompt, column_name):
        # Load the corresponding word2vec model for the given column
        word2vec_model = self.load_word2vec_model(column_name)

        # Compute embedding for the prompt
        prompt_embedding = self.get_word2vec_embedding(prompt, word2vec_model)

        # Extract embeddings from the preprocessed dataframe
        embedding_columns = [col for col in self.preprocessed_data.columns if col.startswith(f'{column_name.upper()}_EMBEDDING_')]

        if not embedding_columns:
            raise ValueError(f"No embeddings found for column: {column_name}")

        # Get the embedding values for each row in the preprocessed dataframe
        embeddings = self.preprocessed_data[embedding_columns].values

        # Compute cosine similarity between prompt embedding and all row embeddings
        similarities = cosine_similarity([prompt_embedding], embeddings)[0]

        # Create result dataframe with the original column and similarity scores
        result_df = pd.DataFrame({
            column_name: self.data[column_name],
            'cosine_similarity': similarities
        })

        return result_df.sort_values(by='cosine_similarity', ascending=False)