from flask import Flask,request,render_template


from src.pipelines.predict_pipeline import CustomData,PredictPipeline

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html') 

@app.route('/predictdata',methods=['POST','GET'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            LIMIT_BAL=int(request.form.get('limit_bal')),
            SEX=int(request.form.get('sex')),
            EDUCATION=int(request.form.get('education')),
            MARRIAGE=int(request.form.get('marriage')),
            AGE=int(request.form.get('age')),
            PAY_0=int(request.form.get('pay_0')),
            PAY_2=int(request.form.get('pay_2')),
            PAY_3=int(request.form.get('pay_3')),
            PAY_4=int(request.form.get('pay_4')),
            PAY_5=int(request.form.get('pay_5')),
            PAY_6=int(request.form.get('pay_6')),
            BILL_AMT1=int(request.form.get('bill_amt1')),
            BILL_AMT2=int(request.form.get('bill_amt2')),
            BILL_AMT3=int(request.form.get('bill_amt3')),
            BILL_AMT4=int(request.form.get('bill_amt4')),
            BILL_AMT5=int(request.form.get('bill_amt5')),
            BILL_AMT6=int(request.form.get('bill_amt6')),
            PAY_AMT1=int(request.form.get('pay_amt1')),
            PAY_AMT2=int(request.form.get('pay_amt2')),
            PAY_AMT3=int(request.form.get('pay_amt3')),
            PAY_AMT4=int(request.form.get('pay_amt4')),
            PAY_AMT5=int(request.form.get('pay_amt5')),
            PAY_AMT6=int(request.form.get('pay_amt6')),
        )
        pred_df=data.get_as_dataframe()
        
        predict_pipeline=PredictPipeline()

        pred=predict_pipeline.predict(pred_df)
        result=""
        if(pred==0):
            result="The Person is not likely to have a default next month"
        else:
            result="The Person is likely to have a default next month"
        return render_template('home.html',result=result)


if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080)