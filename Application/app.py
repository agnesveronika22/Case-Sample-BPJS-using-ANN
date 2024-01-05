from flask import Flask, render_template, request, redirect
import pickle
import sklearn
import numpy as np

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html', insurance_cost=2)

 
@app.route('/predict', methods=['POST','GET'])
def index():
    if request.method == 'POST':
        
        with open('model.pkl','rb') as r:
            model = pickle.load(r)
            
        pstv01 = float(request.form['pstv01'])
        pstv10 = float(request.form['pstv10'])
        pstv14 = float(request.form['pstv14'])
        pnk10 = float(request.form['pnk10'])
        pnk15 = float(request.form['pnk15'])

        datas = np.array((pstv01,pstv10,pstv14,pnk10,pnk15))
        datas = np.reshape(datas,(1,-1))
        
        isLayanan = model.predict(datas)
        
        if isLayanan == 1:
            output = "RITP"
        elif isLayanan == 0:
            output = "RJTP"
        else:
            output = 'PROMOTIF'
        
        return render_template('index.html', finalData=output)
if __name__ == '__main__':
    app.run(debug=True)