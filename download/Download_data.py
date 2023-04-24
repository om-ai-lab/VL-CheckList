# This script and the data is from iCAN: Instance-Centric Attention Network for Human-Object Interaction Detection
# The homepage for iCAN is http://chengao.vision/iCAN/ 
# we use the detection result from iCAN 

# Download files from Google Drive with terminal
# Credit: https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive
# Usage: python google_drive.py FILE_ID DESTINATION_FILENAME
# How to get FILE_ID? Click "get sharable link", then you can find it in the end.

import requests


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
# https://docs.google.com/uc?export=download
    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    #print(response)
    save_response_content(response, destination)


if __name__ == "__main__":
    import sys
    if len(sys.argv) is not 3:
        print("Usage: python google_drive.py drive_file_id destination_file_path")
    else:
        # TAKE ID FROM SHAREABLE LINK
        file_id = sys.argv[1]
        # DESTINATION FILE ON YOUR DISK
        destination = sys.argv[2]
        download_file_from_google_drive(file_id, destination)