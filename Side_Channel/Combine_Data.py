import pandas as pd
import os

import pandas as pd

def combine_data():
    # Define the folder where the CSVs are located
    # folder_path = 'LLaMa2_DB/'# Initialize an empty DataFrame for all data
    folder_path = 'Falcon_DB/One-Token-Vector'# Initialize an empty DataFrame for all data
    all_data = pd.DataFrame()

    first_file = True

    # Initialize a variable for keeping track of the unique index for each file
    file_index = 1

    # Loop through each file in the folder, ensuring sorted order if necessary
    for file in sorted(os.listdir(folder_path)):
        if file.endswith('.csv'):
            # Read the CSV file
            temp_df = pd.read_csv(os.path.join(folder_path, file))

            # Add the file_index to the DataFrame as a new column
            temp_df['File Index'] = file_index

            if first_file:
                # For the first file, simply copy it to all_data
                # Ensure 'File Index' column is added to all_data if it's the first file
                all_data = temp_df
                first_file = False
            else:
                # For subsequent files, align them to the first file's columns including 'File Index'
                aligned_df = temp_df.reindex(columns=all_data.columns)
                all_data = pd.concat([all_data, aligned_df], ignore_index=True)

            # Increment the file index for the next file
            file_index += 1

    # After processing all files, save the combined data to a new CSV
    all_data.to_csv(folder_path[:9] + '_combined_data.csv', index=False)

def check_dups(file_path):
    # Load the data
    df = pd.read_csv(file_path)  # Make sure to put the correct path to your CSV file

    # Find duplicated questions
    duplicated_questions = df[df.duplicated(['Question'], keep=False)]

    # Print out the duplicated questions and their row numbers
    if duplicated_questions.empty:
        print("No duplicated questions found.")
    else:
        for index, row in duplicated_questions.iterrows():
            print(f"Row: {index}, Question: \"{row['Question']}\"")

    # If you want to see the count of how many times each question is duplicated
    question_counts = duplicated_questions['Question'].value_counts()
    for question, count in question_counts.items():
        print(f"Question: \"{question}\" is duplicated {count} times.")

def remove_dups(file_path):
    # Load the data
    df = pd.read_csv(file_path)  # Make sure to put the correct path to your CSV file

    # Drop duplicate questions, keep the first occurrence
    df_cleaned = df.drop_duplicates(subset=['Question'], keep='first')

    # Save the cleaned data to a new CSV file
    df_cleaned.to_csv(file_path, index=False)  # Specify your desired output file path

    print(f"Cleaned CSV saved. Original rows: {len(df)}, Cleaned rows: {len(df_cleaned)}")

file_path = 'LLaMa2_DB/LLaMa2_DB_combined_data.csv'
check_dups(file_path)
remove_dups(file_path)
check_dups(file_path)

