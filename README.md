# Table of contents
1. [Project description](#1-project-description)  
2. [Learnings and results](#2-learnings-and-results)  
    2.1. [Dataset](#21-dataset)  
    2.2. [Training and evaluation results](#22-training-and-evaluation-results)  
    2.3. [Using the model](#23-using-the-model)
3. [Run sample](#3-run-sample)  
    3.1. [Setup](#31-setup)  

# 1. Project description 
[[back to the top]](#table-of-contents)

**Instagram Fake Account Checker** is a web application that uses AutoML to determine the type of Instagram account (bot/scam/real/spam) based on the account characteristics provided. The application is built using Flask and AutoGluon for predictions.

![Альтернативный текст](https://i.imgur.com/2NBJui0.png)

What can you find inside:
- Exploratory data analysis (EDA)
- Data preprocessing
- Using AugoGluon for train ML models
- Web application on Flask
  
<br><br>

# 2. Learnings and results
[[back to the top]](#table-of-contents)

### 2.1. Dataset
This dataset was taken from Kaggle ([Source](https://www.kaggle.com/datasets/manumathewjiss/instagram-multi-class-fake-account-dataset-imfad/data)). 
**Thanks to [Manu Mathew Giss](https://www.kaggle.com/manumathewjiss) for creating and publishing the dataset.**


Description from the original source:
>This dataset was created as part of a research project on detecting fake Instagram accounts. Most of the user profiles were manually added and labeled by our team to ensure accuracy. It includes four account types real, spam, scam, and bot making it suitable for multi-class classification tasks.
The goal is to support students, researchers, and developers in building machine learning models for fake account detection. All data is anonymized and cleaned for safe public use.


You can see the data preprocessing and EDA in the **models_building.ipynb**.
<br><br>
### 2.2. Training and evaluation results
[[back to the top]](#table-of-contents)

- After training models we achieved following leaderboad:

| Model                  | score_test | score_val | eval_metric | pred_time_test | pred_time_val | fit_time | pred_time_test_marginal | pred_time_val_marginal | fit_time_marginal | stack_level | can_infer |
|------------------------|------------|-----------|-------------|----------------|---------------|----------|-------------------------|------------------------|-------------------|-------------|-----------|
| NeuralNetFastAI_BAG_L2 | 0.967094   | 0.972114  | accuracy    | 1.269306       | 1.215871      | 51.406853| 0.172047               | 0.100270              | 9.816978         | 2           | True      | 
| LightGBMXT_BAG_L2      | 0.966401   | 0.971508  | accuracy    | 1.138874       | 1.148042      | 44.323732| 0.041615               | 0.032441              | 2.733858         | 2           | True      |
| WeightedEnsemble_L3    | 0.965362   | 0.972893  | accuracy    | 1.452013       | 1.503689      | 61.033208| 0.002000               | 0.001001              | 0.404261         | 3           | True      |
| CatBoost_BAG_L2        | 0.963976   | 0.968563  | accuracy    | 1.113372       | 1.132618      | 43.887599| 0.016113               | 0.017017              | 2.297724         | 2           | True      |
| LightGBM_BAG_L2        | 0.963976   | 0.971508  | accuracy    | 1.165489       | 1.167830      | 45.213857| 0.068230               | 0.052229              | 3.623983         | 2           | True      |
| RandomForestGini_BAG_L2| 0.961552   | 0.968563  | accuracy    | 1.152212       | 1.308296      | 42.181088| 0.054953               | 0.192696              | 0.591213         | 2           | True      |
| CatBoost_BAG_L1        | 0.961205   | 0.968477  | accuracy    | 0.024046       | 0.012601      | 5.048458 | 0.024046               | 0.012601              | 5.048458         | 1           | True      |
| RandomForestEntr_BAG_L1 | 0.961205   | 0.964406  | accuracy    | 0.056190       | 0.158476      | 0.542565 | 0.056190               | 0.158476              | 0.542565         | 1           | True      | 
| LightGBMXT_BAG_L1      | 0.961205   | 0.966398  | accuracy    | 0.130087       | 0.123167      | 2.271028 | 0.130087               | 0.123167              | 2.271028         | 1           | True      |
| LightGBM_BAG_L1        | 0.960513   | 0.968996  | accuracy    | 0.021005       | 0.016009      | 1.276475 | 0.021005               | 0.016009              | 1.276475         | 1           | True      |
| NeuralNetFastAI_BAG_L1 | 0.960513   | 0.964666  | accuracy    | 0.145279       | 0.080843      | 9.330670 | 0.145279               | 0.080843              | 9.330670         | 1           | True      |
| ExtraTreesEntr_BAG_L1  | 0.960166   | 0.964666  | accuracy    | 0.066072       | 0.185765      | 0.491035 | 0.066072               | 0.185765              | 0.491035         | 1           | True      | 
| XGBoost_BAG_L1         | 0.960166   | 0.968823  | accuracy    | 0.410058       | 0.041559      | 2.404624 | 0.410058               | 0.041559              | 2.404624         | 1           | True      | 
| RandomForestEntr_BAG_L2| 0.960166   | 0.969169  | accuracy    | 1.152009       | 1.300730      | 42.156405| 0.054750               | 0.185129              | 0.566530         | 2           | True      |
| ExtraTreesGini_BAG_L1  | 0.959820   | 0.966312  | accuracy    | 0.063454       | 0.205299      | 0.546277 | 0.063454               | 0.205299              | 0.546277         | 1           | True      |
| NeuralNetTorch_BAG_L1  | 0.959820   | 0.969169  | accuracy    | 0.081951       | 0.054417      | 19.063834| 0.081951               | 0.054417              | 19.063834        | 1           | True      |
| WeightedEnsemble_L2    | 0.959820   | 0.969343  | accuracy    | 0.103956       | 0.079013      | 20.615253| 0.001000               | 0.008587              | 0.274944         | 2           | True      |
| RandomForestGini_BAG_L1| 0.958434   | 0.964406  | accuracy    | 0.057984       | 0.157285      | 0.567107 | 0.057984               | 0.157285              | 0.567107         | 1           | True      | 
| KNeighborsUnif_BAG_L1  | 0.957395   | 0.955226  | accuracy    | 0.021005       | 0.043236      | 0.023767 | 0.021005               | 0.043236              | 0.023767         | 1           | True      |
| KNeighborsDist_BAG_L1  | 0.957049   | 0.955486  | accuracy    | 0.020128       | 0.036945      | 0.024036 | 0.020128               | 0.036945              | 0.024036         | 1           | True      |

- As you can see above, all of the results are very good. 
<br><br>
### 2.3. Using the model 
[[back to the top]](#table-of-contents)

Final model will be used in form of web service on Flask via load file **predictor.pkl**

- The application accepts user input, processes it, makes a prediction, and returns the result. 
- The use of Flask makes it easy to deploy this application as a web service.
- When a user submits a POST request to the /predict endpoint, the script collects the input data from the request. This data includes features such as the number of followers, following, posts, bio, profile picture, external link, and threads.
- The script calculates additional features that need for prediction: following_followers_ratio (ratio of the number of accounts the user is following to the number of followers), posts_followers_ratio(ratio of the number of posts to the number of followers).
- The input data and calculated features are organized into a Pandas DataFrame. This DataFrame is structured in a way that the model can understand and process.
- The script uses the loaded model to make a prediction on the prepared DataFrame. The predictor.predict method is called with predict_df as the input, and it returns a numerical class label.
- The numerical class label returned by the model is mapped to a human-readable class label using the class_mapping dictionary. For example, if the model returns 0, it is mapped to 'Bot'.
- The predicted class label is then returned as a JSON response to the user. This response indicates the type of Instagram account predicted by the model.
  
<br><br>

# 3. Run sample
[[back to the top]](#table-of-contents)

### 3.1. Setup
  - **Download content of this repository**

    You can either clone this repo or just download it and unzip to some folder:

    ```
    git clone https://github.com/syndicate2k/InstagramFakeAccountChecker
    cd InstagramFakeAccountChecker
    ```
 - **Install dependencies**
   
   Install the required libraries using pip:
   ```
   pip install -r requirements.txt
   ```
- **Launch the app**
  
  The application will be available at http://127.0.0.1:5000/ after command:
   ```
   python app.py
   ``` 
- **Instructions for use**
  * Open the app in your browser.
  * Enter your Instagram account details: number of subscribers, followings, posts, as well as the presence of a biography, profile photo, external link and thread.
  * Click the "Predict" button to get a prediction.
  * Enjoy your predictions.


<br><br>
