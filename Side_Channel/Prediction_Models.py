from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.lines as mlines


def train_xgb(X_train, y_train, n_estimators, random_state):
    model = XGBClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def balance_dataset(df, target_column):
    # Find the smallest number of samples for any disease
    min_samples = df[target_column].value_counts().min()

    # Create a balanced DataFrame
    balanced_df = pd.DataFrame()
    for disease in df[target_column].unique():
        disease_df = df[df[target_column] == disease]
        balanced_df = pd.concat([balanced_df, disease_df.sample(min_samples, random_state=25)])

    return balanced_df


def load_and_preprocess_data(balanced_df, with_tokens):
    if with_tokens:
        try:
            X = balanced_df.drop(
                ['File Index', 'Question', 'Answer', 'Words in Question', 'Tokens in Question', 'Words in Answer'],
                axis=1)
            X['AVG1'] = pd.to_numeric(X['AVG1'], errors='coerce')
            X['AVG2'] = pd.to_numeric(X['AVG2'], errors='coerce')
        except:
            X = balanced_df.drop(
                ['File Index', 'Question', 'Answer', 'Words in Question', 'Words in Answer'],
                axis=1)
            X['AVG1'] = pd.to_numeric(X['AVG1'], errors='coerce')
            X['AVG2'] = pd.to_numeric(X['AVG2'], errors='coerce')
    else:
        try:
            X = balanced_df.drop(
                ['File Index', 'Question', 'Answer', 'Words in Question', 'Tokens in Question', 'Tokens in Answer',
                 'AVG1', 'AVG2', 'Words in Answer'],
                axis=1)
        except:
            X = balanced_df.drop(
                ['File Index', 'Question', 'Answer', 'Words in Question', 'Words in Answer',
                 'Tokens in Answer', 'AVG1', 'AVG2'],
                axis=1)

    y = balanced_df['File Index'] - 1  # Adjust class labels to start from 0
    return X, y


def load_and_preprocess_data_disease(df):
    X = df.drop(['Disease', 'Question', 'Answer', 'Words in Question', 'Tokens in Question',
                 'Words in Answer'],
                axis=1)
    y = df['Disease']
    X['AVG1'] = pd.to_numeric(X['AVG1'], errors='coerce')
    X['AVG2'] = pd.to_numeric(X['AVG2'], errors='coerce')
    return X, y


def split_data(X, y, stratify_col):
    # Using stratify parameter to maintain the distribution of 'File Index'
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=24, stratify=stratify_col)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=24, stratify=y_temp)
    return X_train, X_dev, X_test, y_train, y_dev, y_test


def train_nn(X_train, y_train, input_dim, num_classes, epochs, batch_size):
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    dummy_y = to_categorical(encoded_Y, num_classes=num_classes)

    model = Sequential([
        Dense(640, input_dim=input_dim, activation='relu'),
        Dropout(0.5),
        Dense(1280, activation='relu'),
        BatchNormalization(),
        Dense(640, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, dummy_y, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
    return model, encoder


def evaluate_model(model, X_test, y_test, encoder=None):
    # Check if the model is of a type that outputs probabilities
    if hasattr(model, 'predict_proba'):
        y_pred_probs = model.predict_proba(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
    elif hasattr(model, 'predict') and 'Sequential' in str(type(model)):
        # This condition is for handling Keras/TensorFlow models that use 'predict'
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
    else:
        # For models that directly return class labels (e.g., XGBoost)
        y_pred = model.predict(X_test)

    # No need for encoder if y_test is already in correct format
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    return precision, recall, f1, y_pred


def create_and_run_NN(X, y, X_train, X_dev, X_test, y_train, y_dev, y_test):
    epochs_list = [10]
    batch_size_list = [32]

    best_models = []
    for epochs in epochs_list:
        for batch_size in batch_size_list:
            model, encoder = train_nn(X_train, y_train, input_dim=X_train.shape[1], num_classes=len(np.unique(y)),
                                      epochs=epochs, batch_size=batch_size)
            precision, recall, f1, _ = evaluate_model(model, X_dev, y_dev, encoder)
            # Store model, encoder, and its f1 score along with hyperparameters
            best_models.append((model, encoder, epochs, batch_size, f1))
            model.save('Topic_Inference_Falcon_model.h5')

    # Select the best 3 models based on their F1 score on the dev set
    best_models = sorted(best_models, key=lambda x: x[3], reverse=True)[:1]

    # Evaluate the best models on the test set
    for model, encoder, epochs, batch_size, _ in best_models:
        precision, recall, f1, y_pred = evaluate_model(model, X_test, y_test, encoder)
        print(
            f'Epochs: {epochs}, Batch Size: {batch_size}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}')
        cm = confusion_matrix(encoder.transform(y_test), y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title(
            f'NN Confusion Matrix - Epochs: {epochs}, Batch Size: {batch_size}\nPrecision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}')
        plt.savefig('NN_Confusion_Matrix_Falcon_Time_Tokens_AVG.png')

        plt.show()
        return best_models[0][0]


def create_and_run_XGBoost(X, y, X_train, X_dev, X_test, y_train, y_dev, y_test, disease_or_regular):
    # XGBoost hyperparameter tuning
    n_estimators_range = [100, 200, 225, 250, 300, 325, 350, 400]
    seed_range = [23, 24, 25, 26]

    best_xgb_models = []
    for n_estimators in n_estimators_range:
        for random_state in seed_range:
            xgb_model = train_xgb(X_train, y_train, n_estimators, random_state)
            precision, recall, f1, y_pred = evaluate_model(xgb_model, X_dev, y_dev,
                                                           encoder=None)  # Assume evaluate_model is adapted to handle both NN and XGB models
            # Store XGB model and its f1 score
            best_xgb_models.append((xgb_model, n_estimators, random_state, f1))
    # Select the best 3 XGB models based on their F1 score on the dev set
    best_xgb_models = sorted(best_xgb_models, key=lambda x: x[3], reverse=True)[:1]
    if disease_or_regular:
        class_labels = ['General', 'Cancer', 'Dental', 'STD']
    # Define class labels
    else:
        class_labels = ['Code', 'Math', 'Medical', 'Sport', 'Trivia']

    # Evaluate the best XGB models on the test set
    for xgb_model, n_estimators, random_state, _ in best_xgb_models:
        precision, recall, f1, y_pred = evaluate_model(xgb_model, X_test, y_test, encoder=None)  # Adapt as needed
        print(
            f'n_estimators: {n_estimators}, Random State: {random_state}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}')
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(cmap='Blues')
        title = f'XGBoost Confusion Matrix - n_estimators: {n_estimators}, Random State: {random_state}\nPrecision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}'
        plt.title(title)
        plt.savefig('XGBoost_Confusion_Matrix_Falcon_Time_Tokens_AVG.png')
        plt.show()
    return best_xgb_models[0][0]


def plot_and_save_roc_curve(disease_or_regular, y_test, y_pred_proba, title='NN_ROC_Curve_Falcon_Time_Tokens_AVG',):
    if disease_or_regular:
        class_labels = ['General', 'Cancer', 'Dental', 'STD']
    # Define class labels
    else:
        class_labels = ['Code', 'Math', 'Medical', 'Sport', 'Trivia']

    # Binarize the output
    y_test_binarized = label_binarize(y_test, classes=[*range(len(class_labels))])
    n_classes = y_test_binarized.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(class_labels[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.savefig(title + '.png')
    plt.show()


def calculate_tpr_at_fpr_thresholds(model, X_test, y_test, fpr_thresholds=[0.01, 0.001, 0.0001]):
    # Get the predicted probabilities for all classes
    y_scores = model.predict_proba(X_test)

    n_classes = y_scores.shape[1]
    tpr_at_thresholds = {}

    for i in range(n_classes):
        # Get the predicted probabilities for class i
        y_score = y_scores[:, i]

        # Get the true labels for class i
        y_true = (y_test == i).astype(int)

        # Calculate FPR and TPR values for class i
        fpr, tpr, _ = roc_curve(y_true, y_score)

        # Find the TPR values at the specified FPR thresholds
        tpr_values = []
        for threshold in fpr_thresholds:
            idx = np.argmax(fpr > threshold)  # Find the index of the first FPR value larger than the threshold
            if idx > 0:
                tpr_values.append(
                    tpr[idx - 1])  # Append the TPR value corresponding to the FPR value just below the threshold
            else:
                tpr_values.append(0.0)  # If no FPR value is below the threshold, set TPR to 0

        # Store the TPR values for class i
        tpr_at_thresholds[i] = tpr_values

    return tpr_at_thresholds


def plot_tpr_at_fpr_thresholds(model, X_test, y_test, fpr_thresholds=[0.01, 0.001, 0.0001]):
    tpr_values = calculate_tpr_at_fpr_thresholds(model, X_test, y_test, fpr_thresholds)
    class_labels = ['Code', 'Math', 'Medical', 'Sport', 'Trivia']

    plt.figure(figsize=(15, 7))

    num_topics = len(tpr_values.keys())
    offset_step = 0.2  # Adjust this value to control the spacing between columns
    line_thickness = 30  # Adjust this value to control the thickness of the lines

    # Draw a horizontal line at y=0 to emphasize the baseline
    plt.axhline(y=0, color='black', linewidth=0.5, linestyle='--')

    # Plot the vertical lines starting from y=0
    for topic in range(num_topics):
        tpr_list = tpr_values[topic]
        for idx, fpr in enumerate(fpr_thresholds):
            offset = (idx - len(fpr_thresholds) / 2) * offset_step
            x_position = topic + offset
            plt.vlines(x_position, 0, tpr_list[idx], color='C' + str(idx), linestyle='-', linewidth=line_thickness)
            plt.plot(x_position, tpr_list[idx], marker='o', color='C' + str(idx), markersize=5)
            plt.text(x_position, tpr_list[idx] + 0.02, f"{tpr_list[idx]:.2f}", ha='center')

    plt.xticks(range(num_topics), labels=class_labels)
    plt.xlabel('Topics')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('TPR at specific FPR thresholds for each topic')

    legend_markersize = 10  # Size of the markers in the legend
    legend_line_thickness = 10  # Thickness of the lines in the legend

    # Create custom legend handles
    fpr_legend_handles = [
        mlines.Line2D([], [], color='C' + str(idx), marker='o', markersize=legend_markersize,
                      label=f"FPR = {fpr}", linewidth=legend_line_thickness) for idx, fpr in enumerate(fpr_thresholds)
    ]

    # Place the legend at the bottom of the plot with more spacing adjustment
    plt.legend(handles=fpr_legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.1),
               ncol=len(fpr_thresholds), handletextpad=1, frameon=False)

    # Adjust the subplot parameters to give the bottom of the plot more space
    plt.subplots_adjust(bottom=0.3)

    # Save the plot before calling tight_layout to avoid cutting off parts
    plt.savefig("tpr_fpr_adjusted.png")

    # Automatically adjust the subplot params
    plt.tight_layout()

    plt.show()


# Function to append or create CSV with new TPR values
def append_to_csv(df, filepath):
    # Check if the CSV file already exists
    if filepath.is_file():
        df_existing = pd.read_csv(filepath)
        df = pd.concat([df_existing, df], ignore_index=True)
    # Save the dataframe to CSV file
    df.to_csv(filepath, index=False)
    return filepath


def run_tpr_fpr(best_model, X_test, y_test):
    tpr_values = calculate_tpr_at_fpr_thresholds(best_model, X_test, y_test, fpr_thresholds=[0.01, 0.001, 0.0001])
    # Define the class labels
    class_labels = ['Code', 'Math', 'Medical', 'Sport', 'Trivia']
    path_tpr_1 = Path('tpr_fpr1.csv')
    path_tpr_01 = Path('tpr_fpr01.csv')
    path_tpr_001 = Path('tpr_fpr001.csv')
    # Convert TPR values to percentages and round to two decimal places
    for i, tpr_list in tpr_values.items():
        tpr_values[i] = [round(tpr * 100, 2) for tpr in tpr_list]

    # Create three empty dataframes to store TPR values for each FPR threshold
    df_tpr_1 = pd.DataFrame(columns=class_labels)
    df_tpr_01 = pd.DataFrame(columns=class_labels)
    df_tpr_001 = pd.DataFrame(columns=class_labels)

    # Fill the dataframes with TPR values
    for i, tpr_list in tpr_values.items():
        topic_name = class_labels[i]
        df_tpr_1.loc[0, topic_name] = tpr_list[0]
        df_tpr_01.loc[0, topic_name] = tpr_list[1]
        df_tpr_001.loc[0, topic_name] = tpr_list[2]

    # Save the dataframes to CSV files
    append_to_csv(df_tpr_1, path_tpr_1)
    append_to_csv(df_tpr_01, path_tpr_01)
    append_to_csv(df_tpr_001, path_tpr_001)

def run_all_with_different_models(file_path, with_tokens):
    df = pd.read_csv(file_path)

    # # Check class distribution before balancing
    # print("Class distribution before balancing:")
    # print(df['File Index'].value_counts())
    #
    balanced_df = balance_dataset(df, 'File Index')
    #
    # # Check class distribution after balancing
    # print("Class distribution after balancing:")
    # print(balanced_df['File Index'].value_counts())
    # X, y = load_and_preprocess_data(balanced_df)
    X, y = load_and_preprocess_data(balanced_df, with_tokens)
    print(X)
    print(y)

    # Check for NaN values in features
    if X.isnull().values.any():
        print("NaN values found in X after preprocessing")
        X = X.fillna(X.mean())  # Filling NaNs with mean value of each column

    X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(X, y, y)

    # # Check the shapes of the data
    # print(f"Shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")
    # print(f"Shapes: X_dev: {X_dev.shape}, y_dev: {y_dev.shape}")
    # print(f"Shapes: X_test: {X_test.shape}, y_test: {y_test.shape}")

    # best_model = create_and_run_NN(X, y, X_train, X_dev, X_test, y_train, y_dev, y_test)
    best_model = create_and_run_XGBoost(X, y, X_train, X_dev, X_test, y_train, y_dev, y_test,False)

    # Check if the model is predicting only one class
    y_pred = best_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred)
    unique, counts = np.unique(y_pred_classes, return_counts=True)
    print(f"Prediction distribution: {dict(zip(unique, counts))}")
    try:
        y_pred_proba = best_model.predict_proba(X_test)
    except:
        y_pred_proba = best_model.predict(X_test)

    plot_and_save_roc_curve(False, y_test, y_pred_proba)
    # # Convert y_test to DataFrame if it's a Series
    if isinstance(y_test, pd.Series):
        y_test = y_test.to_frame()
    # plot_tpr_at_fpr_thresholds(best_model, X_test, y_test)
    # run_tpr_fpr(best_model, X_test, y_test)


def run_check_disease(file_path):
    df = pd.read_csv(file_path)
    balanced_df = balance_dataset(df, 'Disease')
    X, y = load_and_preprocess_data_disease(balanced_df)
    print(X)
    print(y)

    # Check for NaN values in features
    if X.isnull().values.any():
        print("NaN values found in X after preprocessing")
        X = X.fillna(X.mean())  # Filling NaNs with mean value of each column

    X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(X, y, y)
    best_model = create_and_run_XGBoost(X, y, X_train, X_dev, X_test, y_train, y_dev, y_test,True)

    # Check if the model is predicting only one class
    y_pred = best_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred)
    unique, counts = np.unique(y_pred_classes, return_counts=True)
    print(f"Prediction distribution: {dict(zip(unique, counts))}")
    try:
        y_pred_proba = best_model.predict_proba(X_test)
    except:
        y_pred_proba = best_model.predict(X_test)

    plot_and_save_roc_curve(True, y_test, y_pred_proba)
def main():
    file_path = ['Falcon_DB_combined_data_With_Code.csv']
    # file_path = ['Falcon_DB_combined_data_With_Code.csv', 'LLaMa2_DB_combined_data_With_Code.csv']
    with_tokens = [True]
    for f in file_path:
        for w in with_tokens:
            run_all_with_different_models(f, w)
    # run_check_disease(file_path[0])


if __name__ == "__main__":
    main()
