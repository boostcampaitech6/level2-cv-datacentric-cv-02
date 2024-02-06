import requests
import os
import json
import time
import constants
api_key=constants.API_KEY

fpath = "/data/ephemeral/home/data/medical/img/2001/"
url = "https://api.upstage.ai/v1/document-ai/ocr"
headers = {"Authorization": f"Bearer {api_key}"}
output_dir = "/data/ephemeral/home/data/medical/ufo/2001/"

for imgfile in os.listdir(fpath):
    files = {"image": open(fpath+imgfile, "rb")}
    response = requests.post(url, headers=headers, files=files)
    data = json.loads(response.text.encode('ascii', 'ignore').decode('ascii'))

    output_file = output_dir+imgfile.split('.')[0]+'.json'

    # Write the JSON data to the output file
    with open(output_file, 'w') as file:
        json.dump(data, file)
    time.sleep(0.5)

# limit_list=[]
# for jsonfile in os.listdir(output_dir):
#     tmp_path=output_dir + jsonfile
#     with open(tmp_path, 'r') as file:
#             json_content = file.read()
#     if(json_content == '{"message": "Too Many Requests"}'):
#         limit_list.append(tmp_path.split('/')[-1])

# for item in limit_list:
#     imgfile=item.split('.')[0]+'.jpg'
    
#     files = {"image": open(fpath+imgfile, "rb")}
#     response = requests.post(url, headers=headers, files=files)
#     data = json.loads(response.text.encode('ascii', 'ignore').decode('ascii'))

#     output_file = output_dir+imgfile.split('.')[0]+'.json'

#     # Write the JSON data to the output file
#     with open(output_file, 'w') as file:
#         json.dump(data, file)
    
#     time.sleep(1)

