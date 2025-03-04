import os
import json

def filter_metadata_for_existing_images(json_file_path, image_dir, output_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    existing_images = {img_name for img_name in os.listdir(image_dir) if img_name.endswith('.jpg')}

    filtered_imgs = {img_id: img_data for img_id, img_data in data['imgs'].items()
                     if img_data['file_name'].replace('train/', '') in existing_images}

    filtered_anns = {ann_id: ann_data for ann_id, ann_data in data['anns'].items()
                     if ann_data['image_id'] in filtered_imgs}

    filtered_imgToAnns = {img_id: ann_list for img_id, ann_list in data['imgToAnns'].items()
                          if img_id in filtered_imgs}

    filtered_data = {
        'info': data['info'], 
        'imgs': filtered_imgs,
        'anns': filtered_anns,
        'imgToAnns': filtered_imgToAnns
    }

    with open(output_file_path, 'w') as out_file:
        json.dump(filtered_data, out_file, indent=2)

    print(f"Filtered metadata saved to {output_file_path}")
