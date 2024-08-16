# RCA-Priority-Distribution

RootCausePriorityAnalysis : Analyze and determine priority distribution in root cause analysis using data preprocessing and a machine learning model (RNN-LSTM). Features data cleaning, visualization, and a Flask web interface for user interaction.

## Data Preprocessing

1. **Loading the Dataset**: The dataset is loaded from a CSV file named `jira_priority_cleaned.csv`. The encoding used is `ISO-8859-1`.
2. **Selecting Relevant Columns**: Only the ‘Summary’ and ‘Priority’ columns are selected for further processing.
3. **Encoding the Target Variable**: The ‘Priority’ column is encoded into numerical values using `LabelEncoder` from scikit-learn. This encoded column is named ‘Priority_encoded’.
4. **Text Vectorization**: The text data in the ‘Summary’ column is vectorized using TF-IDF (Term Frequency-Inverse Document Frequency) with a limit of the top 500 features. This converts the text data into numerical form suitable for machine learning models.
5. **Splitting the Dataset**: The dataset is split into training and testing sets using an 80-20 split.

## RNN-LSTM Model

1. **Loading GloVe Embeddings**: GloVe (Global Vectors for Word Representation) embeddings are loaded from a file named `glove.6B.100d.txt`. These embeddings provide pre-trained word vectors that help in capturing semantic meaning.
2. **Tokenizing and Padding Sequences**: The text data is tokenized and converted into sequences of integers. These sequences are then padded to ensure uniform length, which is necessary for input into the RNN-LSTM model.
3. **Preparing the Embedding Matrix**: An embedding matrix is created using the GloVe embeddings. This matrix maps each word in the vocabulary to its corresponding GloVe vector.
4. **Splitting the Dataset**: The dataset is again split into training and testing sets, this time using the tokenized and padded sequences.
5. **Converting Target to Categorical**: The target variable is converted to categorical format for multi-class classification.
6. **Building the RNN-LSTM Model**:
   - An embedding layer is added to the model, initialized with the GloVe embedding matrix.
   - Two bidirectional LSTM layers are added, with dropout layers in between to prevent overfitting.
   - A dense layer with a softmax activation function is added for the final classification.
7. **Compiling the Model**: The model is compiled using the Adam optimizer and categorical cross-entropy loss function.
8. **Training the Model**: The model is trained on the training data for 30 epochs with a batch size of 32. Validation is performed on the testing data.
9. **Evaluating the Model**: The model’s performance is evaluated on the test data, and the results are printed.

## Prediction Function

1. **Preprocessing the Description**:
   - The function `preprocess_description` tokenizes the input text, removes stopwords, and lemmatizes the tokens.
   - It then creates a vector of fixed length (100) by hashing the tokens. This is a simplified approach to convert text into numerical form.
2. **Mapping Predictions to Priority Labels**:
   - A dictionary `priority_mapping` is defined to map numeric predictions to priority labels such as ‘Blocker’, ‘Critical’, ‘Major’, and ‘Minor’.
3. **Example Usage**:
   - An example description is preprocessed using the `preprocess_description` function.
   - The preprocessed data is reshaped to match the model’s expected input shape.
   - The RNN-LSTM model predicts the priority of the issue, and the predicted label is printed.

## Pareto Chart

1. **Loading the Data**:
   - The CSV file `jira_priority_cleaned.csv` is loaded into a DataFrame.
2. **Counting Priority Occurrences**:
   - The occurrences of each priority are counted and sorted in descending order.
3. **Calculating Cumulative Percentage**:
   - The cumulative percentage of the priority counts is calculated.
4. **Creating the Pareto Chart**:
   - A bar plot is created to show the frequency of each priority.
   - A line plot is added to show the cumulative percentage.
   - A horizontal line at 80% is added to highlight the Pareto principle (80/20 rule).
