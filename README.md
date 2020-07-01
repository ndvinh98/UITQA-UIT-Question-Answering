# UITQA-Vietnamese-Question-Answering

A system with a search engine for relevant document retrieval (TF-IDF) and a deep learning model (BERT), applied to answer questions related to regulations of University of Information Technology (UIT)

This project is referenced from [UITHelper_QAS][1] and [Vietnamese question answering with BERT][2]


# To run
1. `pip install -r requirements.txt`

2. Download [my pre-train model][3] and put all file into 'model' directory

3. run cmd `py app.py`

4. go to http://127.0.0.1:5000 


# Train with custom data on BERT question answering
<li>Your dataset must be in SQUADv1.1-like format.

<li>The training and testing data include: Vietnamese question-answer pairs from Wikipedia (same as SQuAD) and Vietnamese question-answer pairs from UIT regulation documents.
  
<li>Your dataset must be in SQUADv1.1 format.

<li>Follow my steps in google_colab/Fine_Tunning_BERT.ipynb


<li>Please note that change the path to your train set and test set


[1]: https://github.com/namnv1113/UITHelper_QAS#general


[2]: https://github.com/mailong25/bert-vietnamese-question-answering

[3]: https://drive.google.com/drive/folders/1MdY-TdDVFdhXQSHU3lPU6oPV3Ndv776V?usp=sharing



