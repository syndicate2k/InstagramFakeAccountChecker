from flask import Flask, request, jsonify, render_template
import pandas as pd
from autogluon.tabular import TabularPredictor

app = Flask(__name__)

models_directory = 'AutogluonModels/ag-20250424_211524/'
predictor = TabularPredictor.load(models_directory)

# for matching one-hot labels with original class labels
class_mapping = {
    0: 'Bot',
    1: 'Scam',
    2: 'Real',
    3: 'Spam'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    followers = int(data.get('followers'))
    following = int(data.get('following'))
    posts = int(data.get('posts'))
    bio = data.get('bio')
    profile_picture = data.get('profile_picture')
    external_link = data.get('external_link')
    threads = data.get('threads')

    # calc some data for prediction
    following_followers_ratio = following / followers if followers > 0 else 0
    posts_followers_ratio = posts / followers if followers > 0 else 0

    predict_df = pd.DataFrame({
        'Followers': followers,
        'Following': following,
        'Following/Followers': following_followers_ratio,
        'Posts': posts,
        'Posts/Followers': posts_followers_ratio,
        'Bio': bio,
        'Profile Picture': profile_picture,
        'External Link': external_link,
        'Threads': threads
    }, index=[0])

    prediction = predictor.predict(predict_df)
    predicted_class = class_mapping[prediction[0]]

    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
