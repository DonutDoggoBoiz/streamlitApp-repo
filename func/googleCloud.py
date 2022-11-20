import streamlit as st
from google.oauth2 import service_account
from google.cloud import storage

#####_CREATE_GCS_API_CLIENT_#####
credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
client = storage.Client(credentials=credentials)
bucket_name = "streamlitapphost.appspot.com"
bucket = client.bucket(bucket_name)

#####_UPLOAD_MODEL_TO_GCS_#####
def upload_model_gcs(save_username, ag_name):
  try:
    local_path = 'model/'+str(save_username)+'_'+str(ag_name)+'.h5'
    gcs_path = 'gcs_model/'+str(save_username)+'/'+str(ag_name)+'.h5'
    gcs_blob = bucket.blob(gcs_path)
    gcs_blob.upload_from_filename(local_path)
    st.success('Upload to GCS DONE!')
  except:
    st.error('ERROR: upload_model_gcs')

#####_DOWNLOAD_MODEL_FROM_GCS_#####
def download_model_gcs(save_username, ag_name):
  try:
    gcs_path = 'gcs_model/'+str(save_username)+'/'+str(ag_name)+'.h5'
    gcs_blob = bucket.blob(gcs_path)
    local_path = 'model/'+str(save_username)+'_'+str(ag_name)+'.h5'
    gcs_blob.download_to_filename(local_path)
    st.success('Download model from GCS DONE!')
  except:
    st.error('ERROR: download_model_gcs')
