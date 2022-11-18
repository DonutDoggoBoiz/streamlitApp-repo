import streamlit as st
from google.oauth2 import service_account
from google.cloud import storage

# Create API client.
credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
client = storage.Client(credentials=credentials)
path_uri = 'gs://streamlitapphost.appspot.com/gcs_mnist_test.csv'
bucket_name = "streamlitapphost.appspot.com"
bucket = client.bucket(bucket_name)

gcs_file_path = 'gcs_mnist_test.csv'

local_path = 'model/'+str(username)+'/'+str(model_name)+'.h5'

content = bucket.blob(file_path).download_to_filename(local_path)

def upload_model(username, model_name):
  local_path = 'model/'+str(username)+'/'+str(model_name)+'.h5'
  gcs_path = 'gcs_model/'+str(username)+'/'+str(model_name)+'.h5'
  gcs_blob = bucket.blob(gcs_path)
  gcs_blob.upload_from_filename(local_path)
