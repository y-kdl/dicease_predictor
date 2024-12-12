# Disease Predictor

The Disease Predictor is a sophisticated machine learning application designed to predict potential medical conditions based on symptoms described by users. It utilizes natural language processing techniques and the K-Nearest Neighbors (KNN) algorithm to analyze and classify symptom data, offering a quick and user-friendly way to gain insights into one's health.

## Features

- **Symptom Analysis**: Users can input their symptoms in natural language. The application processes this text to understand and analyze the user's condition.
- **Disease Prediction**: Implements a K-Nearest Neighbors (KNN) classifier to predict potential diseases based on the processed symptoms.
- **User Interface**: A clean and intuitive interface provided by Streamlit, allowing users to easily interact with the application.
- **Data Processing**: Utilizes TF-IDF vectorization to convert textual symptom data into a format suitable for machine learning.
- **AWS Deployment**: The application is containerized using Docker and deployed on an Amazon EC2 instance, ensuring accessibility and scalability.

## How It Works

1. **Text Preprocessing**: The application first cleans the input symptom text by removing common stopwords and non-informative characters, ensuring that only relevant words are considered for prediction.
2. **Feature Extraction**: It then transforms the preprocessed text into numerical data using TF-IDF vectorization, highlighting the importance of each term in the context of the entire dataset.
3. **Disease Prediction**: The numerical data is fed into a trained KNN classifier, which predicts the most likely disease based on the input symptoms.
4. **Results Presentation**: The predicted disease, along with useful information about the condition and precautions, is displayed to the user.

## Technology Stack

- **Python**: The core language used for the project, leveraging libraries like Pandas, NumPy, Scikit-learn, and NLTK for data processing and machine learning.
- **Streamlit**: For creating a user-friendly web interface.
- **Docker**: For containerizing the application, ensuring consistency across various development and production environments.
- **AWS EC2**: For hosting the application, making it accessible over the internet.
- **Snowflake**: For data storage and management, providing a robust and scalable solution for handling user data.

## Deployment

The application is containerized using Docker and deployed on an Amazon EC2 instance. It's accessible via a public IP address and can handle multiple simultaneous users, showcasing the model's predictions effectively.

Certainly! Here's a revised "Getting Started" section that provides visitors with instructions on how to run the Disease Predictor application:

---

## Getting Started

To get started with the Disease Predictor, follow the steps below to run the application either locally or access it deployed on AWS.

### Running Locally

1. **Clone the Repository**: Clone this repository to your local machine using `git clone <repository-url>`.
2. **Set Up the Environment**:
    - Ensure you have Python 3.9 or later installed.
    - Create a virtual environment: `python -m venv venv`
    - Activate the virtual environment:
        - Windows: `venv\Scripts\activate`
        - MacOS/Linux: `source venv/bin/activate`
    - Install the required dependencies: `pip install -r requirements.txt`
3. **Start the Application**:
    - Run the application using Streamlit: `streamlit run app.py`
    - The application should now be running on `localhost:8501`. Open this address in your web browser to interact with the Disease Predictor.

### Running with Docker

If you have Docker installed, you can run the application using the provided Dockerfile.

1. **Build the Docker Image**: In the project directory, build the image using: `docker build -t streamlit-app .`
2. **Run the Container**: Start the container with: `docker run -p 8501:8501 streamlit-app`
3. **Access the Application**: The application should now be accessible on `localhost:8501`.

### Accessing on AWS

The application is also deployed on an AWS EC2 instance:

1. **Access the Public IP**: Navigate to `http://<Your-EC2-Instance-Public-IP>:8501` in your web browser. Replace `<Your-EC2-Instance-Public-IP>` with the actual public IP address of the EC2 instance where the application is deployed.
2. **Interact with the Application**: Once the page loads, you should see the Disease Predictor interface where you can input symptoms and receive predictions.

### Using the Application

- **Input Symptoms**: Enter your symptoms into the text area provided on the application's interface.
- **Get Predictions**: Submit your symptoms, and the application will display the predicted disease along with relevant details and precautions.

### Notes

- Ensure your firewall and security settings allow traffic on the necessary ports (local: 8501, AWS: as configured).
- If running on AWS, ensure your EC2 instance's security group allows inbound traffic on the port where the application is running.

---

By following these instructions, users should be able to run and interact with the Disease Predictor application either on their local machine or via the deployed version on AWS. Adjust and expand upon these instructions based on the specific details and requirements of your project.

