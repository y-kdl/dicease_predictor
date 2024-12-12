# Importing libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import streamlit as st
from PIL import Image
import snowflake_manipulation as sfm
from datetime import datetime

# Initialize session state variables
if 'want_appointment' not in st.session_state:
    st.session_state['want_appointment'] = False
if 'disease_predicted' not in st.session_state:
    st.session_state['disease_predicted'] = False
if 'predicted_disease' not in st.session_state:
    st.session_state['predicted_disease'] = ""


# Download nltk resources

nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset

data = pd.read_csv('data/Symptom2Disease.csv') 
data.drop(columns=["Unnamed: 0"], inplace=True)

def load_disease_descriptions():
    descriptions_df = pd.read_csv('data/Disease_Descriptions.csv')
    return descriptions_df.set_index('Disease')['Description'].to_dict()

disease_descriptions = load_disease_descriptions()

def load_disease_precaution():
    precaution_df = pd.read_csv('data/Disease_Precautions_Advice.csv')
    return precaution_df.set_index('Disease')['Precaution'].to_dict()

disease_precautions= load_disease_precaution()


# Extracting 'label' and 'text' columns from the 'data' DataFrame

labels = data['label']  
symptoms = data['text'] 


# Text Preprocessing

stop_words = set(stopwords.words('english'))

# Text Preprocessing Function

def preprocess_text(text):
    # Tokenization
    words = word_tokenize(text.lower())
    # Removing stopwords and non-alphabetic characters
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)


# Apply preprocessing to symptoms

preprocessed_symptoms = symptoms.apply(preprocess_text)


# Feature Extraction using TF-IDF

tfidf_vectorizer = TfidfVectorizer(max_features=1500) 
tfidf_features = tfidf_vectorizer.fit_transform(preprocessed_symptoms).toarray()

# Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(tfidf_features, labels, test_size=0.2, random_state=42)

# KNN Model Training

knn_classifier = KNeighborsClassifier(n_neighbors=5) 
knn_classifier.fit(X_train, y_train)


# Streamlit application layout
st.set_page_config(page_title='Disease Symptom Checker', layout='wide')

# Header
st.title('🔍 Disease Symptom Checker')
st.markdown("Welcome to the Disease Symptom Checker. Please enter your symptoms in the text box below and click 'Predict' to see the possible disease.")

# Sidebar - Optional for additional features or information
st.sidebar.header("About the App")
st.sidebar.info("This application is designed to help predict diseases based on symptoms entered by the user. It uses machine learning algorithms to analyze the symptoms and provide a possible diagnosis. This still be a prediction and not a final result. You may need further tests with a specialist.")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("Enter Your Symptoms")
    user_input = st.text_area("", height=150)

with col2:
    st.header("Possible Disease")
    predict_button = st.button('Predict Disease')
    
    if predict_button or st.session_state['disease_predicted']:
        st.session_state['disease_predicted'] = True
        
        if user_input  == ""  : 
            st.info('Enter your symptoms please')
        else:    
            # Preprocess and vectorize user input only if the predict button was pressed
            if predict_button:
                preprocessed_input = preprocess_text(user_input)
                input_vectorized = tfidf_vectorizer.transform([preprocessed_input])
                predicted_disease = knn_classifier.predict(input_vectorized)
                st.session_state['predicted_disease'] = predicted_disease[0]
                
            st.success(f'Based on the symptoms, you may have: {st.session_state["predicted_disease"]}')
            
            # Display the disease description
            if st.session_state["predicted_disease"] in disease_descriptions and st.session_state["predicted_disease"] in disease_precautions:
                st.subheader('Description of the disease : ')
                st.info(disease_descriptions[st.session_state["predicted_disease"]])
                st.subheader('Precaution and Advice')
                st.info(disease_precautions[st.session_state["predicted_disease"]])
            else:
                st.error("No description or precaution available for this disease.")
            
            appointment_button = st.button("Get an appointment with a specialist?")
            if appointment_button:
                st.session_state['want_appointment'] = True
            
            decline_button = st.button("No, thanks")
            if decline_button:
                st.session_state['want_appointment'] = False
                st.session_state['disease_predicted'] = False
                st.session_state['predicted_disease'] = ""
                st.success("Wish you a good recovery! 🌟")
            if st.button("Return to Start"):
               st.experimental_rerun()    
                
    
    else:
        st.write("Your predicted disease and information will appear here.")                
    


if st.session_state['want_appointment']:
    st.subheader("Appointment Details")

    # Get specialists list
    specialists_df = pd.read_csv('data/Medical_Specialists.csv')
    filtered_specialists = specialists_df[specialists_df['Speciality'] == st.session_state['predicted_disease']]
    specialists_list = filtered_specialists["Name"].tolist()
    
    # Display specialists in cards
    for index, specialist in filtered_specialists.iterrows():
        with st.container():
            st.markdown(f"""
                <style>
                .card {{
                    margin: 10px;
                    padding: 10px;
                    border-radius: 10px;
                    border: 1px solid #E1E1E1;
                    box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.1);
                }}
                </style>
                <div class="card">
                    <h4>{specialist["Name"]}</h4>
                    <p><b>Speciality:</b> {specialist["Speciality"]}</p>
                    <p><b>Office Address:</b> {specialist["Office Address"]}</p>
                    <p><b>Phone Number:</b> {specialist["Phone Number"]}</p>
                    <p><b>Years of Experience:</b> {specialist["Years of Experience"]}</p>
                    <p><b>Graduated from:</b> {specialist["Med School"]}</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Select a specialist
    selected_specialist = st.selectbox("Select a Specialist", specialists_list)
    
    # Patient details input
    patient_name = st.text_input("Your Name")
    patient_age = st.number_input("Your Age", step=1, min_value=0)
    patient_address = st.text_input("Your Address")
    patient_email = st.text_input("Your Email")
    patient_phone = st.text_input("Your Phone Number")
    
    # Appointment date
    appointment_date = st.date_input("Appointment Date", min_value=datetime.today())
    
    # Confirm button
    if st.button("Confirm Appointment"):
        #print("Confirm button pressed")
        
        # Prepare patient data
        patient_data = (patient_name, patient_age, patient_address, patient_email, st.session_state['predicted_disease'], patient_phone)
        #print("Patient data:", patient_data)

        # Insert patient data and retrieve ID
        conn = sfm.connect_to_snowflake()
        sfm.insert_patient_data(patient_data)
        patient_id = sfm.retrieve_patient_id(patient_email)
        
        # Prepare and insert appointment data
        if patient_id:
            appointment_data = (patient_id, appointment_date, selected_specialist)
            sfm.insert_appointment_data(appointment_data)
            st.success(f"Appointment confirmed with {selected_specialist}")
        else:
            st.error("Error: Patient ID not found")




# Footer
st.markdown("---")
st.markdown("© 2023 Disease Symptom Checker. By Akram. All Rights Reserved.")
