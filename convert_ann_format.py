import json
import os
from PIL import Image
from argparse import ArgumentParser

ignore_tags = ['masked', 'maintable', 'stamp', 'excluded-region']


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--mode', type=str,
                        default="ufo2datum",
                        help="ufo2datum or datum2ufo")

    args = parser.parse_args()
    return args
    
    

def ufo2datum():
    # Step 1: Read images and get their sizes
    image_folder = '../data/medical/img/train' 
    image_sizes = {}
    for image_name in os.listdir(image_folder):
        if image_name.endswith(('.png', '.jpg', '.jpeg')):
            with Image.open(os.path.join(image_folder, image_name)) as img:
                width, height = img.size
                image_sizes[image_name] = {'width': width, 'height': height}

    # Step 2: Read UFO JSON file
    with open('../data/medical/ufo/train.json', 'r') as file:  
        ufo_data = json.load(file)

    # Step 3: Convert to datumaro format
    datum_data = {
    "info": {},
    "categories": {
        "label": {
        "labels": [
            {
            "name": "1",
            "parent": "",
            "attributes": []
            },
            {
            "name": "2",
            "parent": "",
            "attributes": ["illegibility=True", "maintable=False"]
            },
            {
            "name": "3",
            "parent": "",
            "attributes": ["masked"]
            },
            {
            "name": "4",
            "parent": "",
            "attributes": ["maintable"]
            },
            {
            "name": "5",
            "parent": "",
            "attributes": ["stamp"]
            },
            {
            "name": "6",
            "parent": "",
            "attributes": ["excluded_region"]
            }
        ],
        "attributes": []
        }
    },
    "items": [
    ]
    }
    
    image_id = 0
    annotation_id = 0
    for image_name, size in image_sizes.items():
        image_id_name=image_name[:-4]
        
        datum_image = {
            "id": image_id_name,#image_id,
            "annotations": [],
            "image": {
                    "path": image_name,
                    "size": [
                        size['width'],
                        size['height']
                        
                    ]
                }
        }

        if image_name in ufo_data['images']:
            for word_id, word_info in ufo_data['images'][image_name]['words'].items():                
                # Convert points to Datumaro polygon format [x1, y1, x2, y2, x3, y3, x4, y4]
                points = word_info['points']

                ig_tag=[el for el in word_info['tags'] if el in ignore_tags]
                if(ig_tag):    
                    label=ignore_tags.index(ig_tag[0])+2
                elif(word_info["illegibility"]):
                    label=1
                else:
                    label=0

                datum_annotation = {
                    "id": word_id,
                    "type": "polygon",
                    "attributes": {},
                    "group": 0,
                    "label_id": label,
                    "points": [el for p in points for el in p],
                    "z_order": 0
                }
                datum_image["annotations"].append(datum_annotation)
                annotation_id += 1

        datum_data["items"].append(datum_image)
        image_id += 1

    # Write the Daturamo format JSON to a file
    with open('../datum1.json', 'w') as outfile:
        json.dump(datum_data, outfile, indent=4)
        
        
#pre: need to have input datum json file 'default.json' under data/medical/datumaro
def datum2ufo():             
    with open('../data/medical/datumaro/default.json', 'r') as file:  
            datum_data = json.load(file)

    items=datum_data["items"]

    images=dict()

    for image in items:
        words=dict()
        cnt=1
        for annotation in image["annotations"]:
            temp={
                "points":[],
                "tags":[],
                "illegibility": False
            }
            
            label = annotation["label_id"]
            if(annotation["label_id"]==0):
                pass
            elif(annotation["label_id"]==1):
                temp["illegibility"]= True
            elif(annotation["label_id"]<4):
                temp["illegibility"]= True
                temp["tags"].append(ignore_tags[label-2])
            else:
                temp["illegibility"]= False
                temp["tags"].append(ignore_tags[label-2])
                
            if(annotation["type"]=="bbox"): #bbox
                temp["points"]=[[annotation["bbox"][0],annotation["bbox"][1]],
                                [annotation["bbox"][0]+annotation["bbox"][2],annotation["bbox"][1]],
                                [annotation["bbox"][0],annotation["bbox"][1]+annotation["bbox"][3]],
                                [annotation["bbox"][0]+annotation["bbox"][2],annotation["bbox"][1]+annotation["bbox"][3]],              
                ]
            else:#polygon
                for i in range(0,len(annotation["points"]),2):
                    temp["points"].append([annotation["points"][i],annotation["points"][i+1]])

            
            words[cnt]=temp
            cnt+=1
            
        if("/" in image["id"]):
            image_name=image["id"].split("/")[-1]+".jpg"
        else:
            image_name=image["id"]+".jpg"
        images[image_name]={"words":words}
        
        with open('../ufo1.json', 'w') as outfile:
            final={"images":images}
            json.dump(final, outfile, indent=4)




if __name__ == '__main__':
    args = parse_args()
    if(args.mode=="ufo2datum"):
        ufo2datum()
    elif(args.mode=="datum2ufo"):
        datum2ufo()