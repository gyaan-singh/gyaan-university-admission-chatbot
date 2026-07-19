# Gyaan University Admission Assistant Chatbot

## 📌 Project Overview

The **Gyaan University Admission Assistant Chatbot** is an AI-based chatbot developed using **Python, TensorFlow, and Natural Language Processing (NLP)**.

It helps students get quick answers related to:

- University admissions
- Courses
- Required documents
- Eligibility criteria
- Fees
- Hostel facilities
- General admission queries

The chatbot uses a **deep learning intent classification model** to understand student questions and provide appropriate responses.

---

## 🚀 Features

- Greeting and conversation handling
- Course information assistance
- Admission process guidance
- Required document information
- Eligibility criteria support
- Fee-related queries
- Hostel facility information
- Thank-you and goodbye handling
- Unknown query detection using confidence score

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Natural Language Processing (NLP)
- Tokenization
- Word Embeddings
- Neural Networks

---

## 🧠 Model Architecture

The chatbot uses a neural network model consisting of:

### 1. Embedding Layer

Converts words into numerical vector representations.

### 2. Global Average Pooling Layer

Reduces sequence information into fixed-size features.

### 3. Dense Layer

Learns patterns from training data.

### 4. Softmax Output Layer

Predicts the user's intent category.

### Model Flow

```text
User Input
    |
Tokenization
    |
Sequence Padding
    |
Embedding Layer
    |
Neural Network
    |
Intent Prediction
    |
Response Generation




