
import os
import pandas as pd
from nltk.tokenize import word_tokenize
def remove_words_before_str1(st1, st2):
    st1 = st1[:-2].strip() + "?"
    if isinstance(st2, float):
        return st2, False
    st2_words = str(st2).split()
    st1_words = st1.split()
    try:
        # Find the index of the first word of st1 in st2_words
        start_index = st2_words.index(st1_words[0])
        # Check if the subsequent words match the rest of st1
        for i, word in enumerate(st1_words[1:], start=1):
            if st2_words[start_index + i] != word:
                raise ValueError
        # If all words match, remove everything before and including st1
        return ' '.join(st2_words[start_index + len(st1_words):]), True
    except ValueError:
        return st2, False

def process_csv(file_path):
    # Load the DataFrame from the CSV file
    df = pd.read_csv(file_path)

    # Apply the remove_words_before_str1 function to each row
    answers, modified = zip(*df.apply(lambda row: remove_words_before_str1(row['Question'], row['Answer']), axis=1))
    modified = list(modified)  # Convert the Boolean array to a list for indexing
    df['Answer'] = answers  # Update the 'Answer' column with modified answers
    df['Tokens in Answer'] = df.apply(lambda row: len(word_tokenize(str(row['Answer']))), axis=1)
    df['Tokens in Question'] = df.apply(lambda row: len(word_tokenize(row['Question'])), axis=1)
    # df.loc[modified, 'Tokens in Answer'] -= df.loc[modified, 'Tokens in Question']
    df.drop(columns=['Tokens in Question'], inplace=True)

    # Save the modified DataFrame back to the CSV file
    df.to_csv(file_path, index=False)

def combine_data():
    # Define the folder where the CSVs are located
    folder_path = 'LLaMa2_DB/'# Initialize an empty DataFrame for all data
    # folder_path = 'Falcon_DB'  # Initialize an empty DataFrame for all data
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
    all_data.to_csv(folder_path[:9] + '_combined_data_With_Code.csv', index=False)


def check_dups(file_path):
    # Load the data
    df = pd.read_csv(file_path)  # Make sure to put the correct path to your CSV file

    # Find duplicated questions
    duplicated_questions = df[df.duplicated(['Question'], keep=False)]

    # Print out the duplicated questions and their row numbers
    if duplicated_questions.empty:
        print("No duplicated questions found.")

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


# Read the CSV file

# Function to check if 'cancer' is in the question or answer
def check_disease(row):
    question = row['Question']
    answer = row['Answer']

    if 'cancer' in question or 'cancer' in answer:
        return 1
    elif any(word in question or word in answer for word in ['teeth', 'dentist', 'tooth',"cavity","enamel","braces","flossing","plaque"]):
        return 2
    elif any(word in question or word in answer for word in
             ['aids', 'hiv', 'hiv-1', 'hiv-2', 'elisa', 'genital', "herpes","syphilis","gonorrhea","chlamydia","papillomavirus","hpv","condom","hepatitis"]):
        return 3
    else:
        return 0


def sort_diseases(file_path):
    df = pd.read_csv(file_path)  # Default separator is comma
    # Apply the function to each row and create the 'Category' column
    df['Disease'] = df.apply(check_disease, axis=1)

    # Save the new CSV file with the 'Category' column
    df.to_csv(file_path[:-4] + '_Diseases.csv', index=False)

    # Count the number of rows in each category
    category_counts = df['Disease'].value_counts()

    # Print the counts
    for category, count in category_counts.items():
        print(f'Number of rows in category {category}: {count}')


def remove_zeros_answer(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    # Filter out the rows where 'Tokens in Answer' is 0 or 1
    filtered_df = df[df['Tokens in Answer'] > 1]
    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(file_path, index=False)

def main():
    file_path = 'Falcon_DB_combined_data_With_Code.csv'
    check_dups(file_path)
    remove_dups(file_path)
    check_dups(file_path)
    remove_zeros_answer(file_path)
    # process_csv(file_path)
    # sort_diseases(file_path)
    # combine_data()


if __name__ == '__main__':
    main()
