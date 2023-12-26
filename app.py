from flask import Flask, render_template, request
from azure.storage.blob import BlobServiceClient
import os
import numpy as np
from joblib import load

app = Flask(__name__)

# Azure Blob Storage Configuration
account_name = 'loanstore'
account_key = 'xzdpfLxhgChH1Id0PxKKw+noCONjNqpDHwfWXTSMTMBWSpUG28kRimtXmb1RyUFMFocIxR+Iav8q+AStUD3s6Q=='
container_name = 'joblibstore'
blob_name = 'file.joblib'
downloaded_model_path = 'downloads/model.joblib'

# Download the model from Azure Blob Storage
connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)
blob_client = container_client.get_blob_client(blob_name)

with open(downloaded_model_path, "wb") as my_blob:
    download_stream = blob_client.download_blob()
    my_blob.write(download_stream.readall())

# Load the model
new_loanapp = load(downloaded_model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        refno = request.form.to_dict()
        refno = list(refno.values())
        refno = list(map(int, refno))
        refno = np.array(refno).reshape(1, 11)
        result = new_loanapp.predict(refno)
        if int(result) == 1:
            prediction = 'Congratulations! Your Loan is Approved'
        else:
            prediction = 'Sorry! Your Loan Application Is REJECTED'
        return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
