from flask import Flask, render_template, request, jsonify
from src.pipeline.prediction_pipeline import PredictPipeline,CustomData


app = Flask(__name__)

application=app

@app.route('/',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            carat=float(request.form.get('carat')),
            depth=float(request.form.get('depth')),
            table=float(request.form.get('table')),
            x=float(request.form.get('x')),
            y=float(request.form.get('y')),
            z=float(request.form.get('z')),
            cut=request.form.get('cut'),
            color=request.form.get('color'),
            clarity=request.form.get('clarity')
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        result=round(pred[0],2)
        return render_template('form.html',result=result)
    

if __name__ == '__main__':
    app.run(debug=True)