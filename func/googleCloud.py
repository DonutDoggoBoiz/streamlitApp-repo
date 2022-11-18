import streamlit as st
from google.oauth2 import service_account
from google.cloud import storage

# Create API client.
credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
client = storage.Client(credentials=credentials)
path_uri = 'gs://streamlitapphost.appspot.com/gcs_mnist_test.csv'
bucket_name = "streamlitapphost.appspot.com"
bucket = client.bucket(bucket_name)

# upload function
def upload_model(save_username, ag_name):
  local_path = 'model/'+str(save_username)+'/'+str(ag_name)+'.h5'
  gcs_path = 'gcs_model/'+str(save_username)+'/'+str(ag_name)+'.h5'
  gcs_blob = bucket.blob(gcs_path)
  gcs_blob.upload_from_filename(local_path)
