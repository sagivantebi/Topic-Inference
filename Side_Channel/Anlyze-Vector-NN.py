import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Embedding, Dropout, Dense, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Load the data
df = pd.read_csv('Falcon_DB_combined_data.csv')  # Make sure to put the correct path to your CSV file
df['Token times vector'] = df['Token times vector'].apply(lambda x: eval(x))

# Find the length of the longest sequence
max_sequence_length = df['Token times vector'].apply(len).max()
print(max_sequence_length)

# Pad sequences
X_token_times = pad_sequences(df['Token times vector'].values.tolist(), maxlen=max_sequence_length, padding='post')

# Standardize other columns
scaler = StandardScaler()
X_tokens_time = scaler.fit_transform(df[['Tokens in Answer', 'Time to Generate']].values)

# One-hot encode the target variable
y = to_categorical(df['File Index'].values)

# Split the data with stratification
X_combined = np.hstack((X_token_times, X_tokens_time))
X_tt_train, X_tt_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=25, stratify=df['File Index'])

# After splitting, you need to separate the token times and tokens & time features again
X_ttv_train = X_tt_train[:, :max_sequence_length]
X_ttv_test = X_tt_test[:, :max_sequence_length]
X_tt_train = X_tt_train[:, max_sequence_length:]
X_tt_test = X_tt_test[:, max_sequence_length:]

# Define the RNN model
input_token_times = Input(shape=(max_sequence_length,))
input_tokens_time = Input(shape=(2,))

# LSTM branch
x = Embedding(input_dim=max_sequence_length + 1, output_dim=64)(input_token_times)
x = LSTM(64)(x)
x = Model(inputs=input_token_times, outputs=x)

# ANN branch
y = Dense(64, activation='relu')(input_tokens_time)
y = Dense(64, activation='relu')(y)
y = Model(inputs=input_tokens_time, outputs=y)

# Combine RNN and ANN branches
combined = concatenate([x.output, y.output])

# Fully connected layers
z = Dense(64, activation='relu')(combined)
z = Dropout(0.5)(z)
z = Dense(32, activation='relu')(z)
z = Dense(y_train.shape[1], activation='softmax')(z)

# Final model
model = Model(inputs=[x.input, y.input], outputs=z)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([X_ttv_train, X_tt_train], y_train, epochs=25, batch_size=32, verbose=0)
# You can adjust epochs and batch_size as needed

# Predict on test set
y_pred = model.predict([X_ttv_test, X_tt_test])
y_pred = y_pred.argmax(axis=1)
y_test_argmax = y_test.argmax(axis=1)

# Calculate metrics
precision = precision_score(y_test_argmax, y_pred, average='macro')
recall = recall_score(y_test_argmax, y_pred, average='macro')
f1 = f1_score(y_test_argmax, y_pred, average='macro')

print(f'Precision: {precision}, Recall: {recall}, F1-Score: {f1}')

# Confusion matrix
cm = confusion_matrix(y_test_argmax, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title(f'Confusion Matrix\nPrecision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}')
plt.savefig('confusion_matrix.png')  # This will save the confusion matrix plot as a PNG file
plt.show()

# Save the model
model.save('Topic_Inference_LSTM_model.h5')

# To load the model later, you can use
# loaded_model = load_model('Topic_Inference_LSTM_model.h5')
