from flask import Flask, render_template, request, redirect, url_for
import os
from forms import *
from IR import *
from bert import QA
import time

basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

MODEL_PATH=os.path.join(basedir, 'model')
DATA_PATH=os.path.join(basedir, 'dataUIT')
data=load_data(DATA_PATH)
stopwords = set(open(basedir+'\stopwords.txt',encoding="utf-8").read().split(' ')[:-1])

#load model
print("Load model...")
start = time.time()
model=QA(MODEL_PATH) #path to model
end = time.time()
print("time load model: "+str(round((end - start),2)))

#Building index
print('Building index...')
start = time.time()
data_standard=standardize_data(data,stopwords)
vect = TfidfVectorizer(min_df=1, max_df=0.8,max_features=5000,sublinear_tf=True,ngram_range=(1,3)) 
vect.fit(data_standard)
end = time.time()
print("Time building index: "+str(round((end - start),2)))



@app.route('/', methods=['GET','POST'])
def home():
    return redirect(url_for('index'))
    #return render_template('index.html',form=form)

@app.route('/index', methods=['GET', 'POST'])
def index():
    form=SearchForm(request.form)
    query = form.data['search']
    if query != '':
        start = time.time()
        results=IR_QA(query=query,data=data,model=model,tf_idf_vetor=vect,top_n_matching=4)
        end = time.time()
        time_processing=round((end - start),2)
        if results:
            return render_template('index.html', form=form, query=query, results=results, time= time_processing)
    return render_template('index.html', form=form)



# @app.after_request
# def add_header(r):
#     """
#     Add headers to both force latest IE rendering engine or Chrome Frame,
#     and also to cache the rendered page for 10 minutes.
#     """
#     r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
#     r.headers["Pragma"] = "no-cache"
#     r.headers["Expires"] = "0"
#     r.headers['Cache-Control'] = 'public, max-age=0'
#     return r


if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)
    TEMPLATES_AUTO_RELOAD = True