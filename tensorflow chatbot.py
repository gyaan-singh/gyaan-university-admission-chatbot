import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# -----------------------------
# Training Data (Gyaan University)
# -----------------------------
sentences = [
    "hello",
    "hi",
    "good morning",

    "what courses are available",
    "which programs do you offer",
    "tell me about courses",

    "how can i apply for admission",
    "admission process",
    "how to take admission",

    "what documents are required",
    "which documents do i need",

    "what is eligibility criteria",
    "who can apply",

    "what is fee structure",
    "tell me about fees",

    "is hostel available",
    "do you provide hostel",

    "thank you",
    "thanks",

    "bye",
    "goodbye"
]

# Labels:
# 0 Greeting
# 1 Courses
# 2 Admission Process
# 3 Documents
# 4 Eligibility
# 5 Fees
# 6 Hostel
# 7 Thanks
# 8 Goodbye

labels = [
    0,0,0,
    1,1,1,
    2,2,2,
    3,3,
    4,4,
    5,5,
    6,6,
    7,7,
    8,8
]

responses = {
    0: "Hello! Welcome to Gyaan University Admission Assistant. How can I help you?",
    1: "Gyaan University offers various undergraduate and postgraduate programs. Please tell me the course you are interested in.",
    2: "You can apply for admission through the university admission process. Please check the admission portal for application details.",
    3: "Required documents generally include academic certificates, ID proof, photographs, and application forms.",
    4: "Eligibility depends on the selected program. Please mention the course name for specific eligibility details.",
    5: "Fee structure depends on the course and program. Please specify your course for detailed fee information.",
    6: "Hostel facilities are available. Please contact the university admission office for hostel details.",
    7: "You're welcome! Feel free to ask any admission-related questions.",
    8: "Thank you for contacting Gyaan University Admission Assistant. Goodbye!"
}

# -----------------------------
# Tokenization
# -----------------------------
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)

max_length = 6

padded = pad_sequences(
    sequences,
    maxlen=max_length,
    padding="post"
)

# -----------------------------
# Build TensorFlow Model
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=1000,
        output_dim=16
    ),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        9,
        activation="softmax"
    )
])

# -----------------------------
# Compile Model
# -----------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# Train Model
# -----------------------------
model.fit(
    padded,
    np.array(labels),
    epochs=200,
    verbose=0
)

print("Gyaan University Admission Chatbot is ready!")

# -----------------------------
# Chat Interface
# -----------------------------
while True:
    user_input = input("\nStudent: ")

    if user_input.lower() == "exit":
        print("Bot: Thank you for visiting Gyaan University.")
        break

    sequence = tokenizer.texts_to_sequences([user_input])

    padded_input = pad_sequences(
        sequence,
        maxlen=max_length,
        padding="post"
    )

    prediction = model.predict(
        padded_input,
        verbose=0
    )

    intent = np.argmax(prediction)
    confidence = np.max(prediction)

    if confidence < 0.5:
        print("Bot: Sorry, I didn't understand. Please contact the admission office for more details.")
    else:
        print("Bot:", responses[intent])
