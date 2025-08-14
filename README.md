
# HP Printer -Product Reviews Classification

This project aims to classify the sentiment of HP printer reviews into Positive, Neutral, and Negative categories using natural language processing techniques and deep learning models.

## How to Run the Notebook

1.  **Open in Google Colab or VSCode:** Upload the notebook file (`.ipynb`) to your Google Drive and open it with Google Colab or in local using vscode application or using jupyter .
2.  **Run Cells:** Execute the code cells sequentially from top to bottom.
    *   Ensure you have a stable internet connection to download necessary libraries and models.
    *   The notebook includes cells for installing required libraries (`nltk`, `fasttext`, `imblearn`, `transformers`, `torch`). These will be installed automatically when you run the respective cells.
3.  **Interact:** Some cells will require user interaction, such as entering a review for sentiment prediction. Follow the prompts in the output.


4. **Load the data correctly to the notebook**

`Datasets`: There are two Datasets in the files, one is orginal dataset and another one is argumented dataset.  
**Orginal dataset**:`HP_Wireless_Printer_Reviews.csv`
**Argumented dataset**:`hp_printer_reviews_5000.csv`. 
Use argumented dataset.


## Libraries Used (Dependencies)

The following Python libraries are used in this notebook:

*   `pandas`: For data manipulation and analysis.
*   `numpy`: For numerical operations.
*   `matplotlib.pyplot`: For data visualization.
*   `seaborn`: For enhanced data visualization.
*   `re`: For regular expressions (text cleaning).
*   `nltk`: For natural language processing tasks (tokenization, stopwords, lemmatization, POS tagging).
*   `collections.Counter`: For word frequency analysis.
*   `sklearn`: For data splitting, evaluation metrics (classification_report, confusion_matrix, roc_auc_score, compute_class_weight), and label encoding.
*   `imblearn.over_sampling.BorderlineSMOTE`: For handling class imbalance.
*   `torch`: For building and training deep learning models (LSTM, GRU, BiLSTM, BERT).
*   `torch.nn`: For neural network modules.
*   `torch.optim`: For optimization algorithms.
*   `torch.utils.data`: For creating Datasets and DataLoaders.
*   `transformers`: For loading and using pre-trained BERT models and tokenizers.
*   `pickle`: For saving and loading models and other objects.


## Project Structure (within the notebook)

The notebook is structured into the following sections:

1.  **Library Imports:** Importing necessary Python libraries.
2.  **Data Loading and Initial Inspection:** Loading the dataset and displaying initial information.
3.  **Data Cleaning and Preprocessing:** Steps for cleaning review text (lowercase, punctuation removal, tokenization, stopword removal, lemmatization).
4.  **Exploratory Data Analysis (EDA):** Analyzing word frequencies and review lengths.
5.  **Rule-Based Sentiment Analysis:** Implementing a basic rule-based approach as a baseline.
6.  **FastText Embedding and Traditional RNN Models:** Generating FastText embeddings, handling class imbalance, defining and training LSTM, GRU, and BiLSTM models.
7.  **BERT-based Sentiment Classification:** Loading BERT, preparing data for BERT, defining and fine-tuning a BERT classifier.
8.  **Model Evaluation:** Evaluating the performance of the trained models using various metrics.
9.  **Model Saving and Prediction:** Saving the best performing model and demonstrating how to make predictions on new reviews.
## 
Gmail: 
chintabalasaireddy@gmail.com