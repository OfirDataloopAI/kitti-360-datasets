import dtlpy as dl
import uuid
import os
import shutil
import json
from io import BytesIO


def update_frames_item(frames_item: dl.Item):
    dataset = frames_item.dataset
    uid = str(uuid.uuid4())
    base_path = "{}_{}".format(dataset.name, uid)
    try:
        os.makedirs(name=base_path, exist_ok=True)
        items_download_path = os.path.join(os.getcwd(), base_path)
        print("Downloading jsons...")
        dataset.download_annotations(
            local_path=items_download_path,
            annotation_options=dl.ViewAnnotationOptions.JSON
        )
        print("Download Completed! Updating 'frames.json'...")
        jsons_path = os.path.join(items_download_path, "json")
        buffer = frames_item.download(save_locally=False)
        frames_json = json.load(fp=buffer)

        # Update frames
        for pcd_idx, frame in enumerate(frames_json.get("frames", list())):
            pcd_relative_filepath = frame["lidar"]["remote_path"][1:]
            pcd_json_relative_filepath = f"{os.path.splitext(pcd_relative_filepath)[0]}.json"
            pcd_filepath = os.path.join(jsons_path, frames_item.dir[1:], pcd_json_relative_filepath)

            with open(pcd_filepath, 'r') as pcd_json_file:
                pcd_json = json.load(fp=pcd_json_file)
            pcd_id = pcd_json["id"]
            frames_json["frames"][pcd_idx]["lidar"]["lidar_pcd_id"] = pcd_id
            frames_json["frames"][pcd_idx]['groundMapId'] = None

            # Update PCD remote_path
            if frames_item.dir == "/":
                pcd_remote_path = f"{frames_item.dir}{pcd_relative_filepath}"
            else:
                pcd_remote_path = f"{frames_item.dir}/{pcd_relative_filepath}"
            frames_json["frames"][pcd_idx]["lidar"]["remote_path"] = pcd_remote_path

            for image_idx, image in enumerate(frame.get("images", list())):
                image_relative_path = image["remote_path"][1:]
                image_json_relative_filepath = f"{os.path.splitext(image_relative_path)[0]}.json"
                image_filepath = os.path.join(jsons_path, frames_item.dir[1:], image_json_relative_filepath)

                with open(image_filepath, 'r') as image_json_file:
                    image_json = json.load(fp=image_json_file)
                image_id = image_json["id"]
                frames_json["frames"][pcd_idx]["images"][image_idx]["image_id"] = image_id

                # Update Image remote_path
                if frames_item.dir == "/":
                    image_remote_path = f"{frames_item.dir}{image_relative_path}"
                else:
                    image_remote_path = f"{frames_item.dir}/{image_relative_path}"
                frames_json["frames"][pcd_idx]["images"][image_idx]["remote_path"] = image_remote_path

        buffer = BytesIO()
        buffer.write(json.dumps(frames_json, default=lambda x: None).encode())
        buffer.seek(0)
        buffer.name = "frames.json"
        frames_item = dataset.items.upload(
            remote_path=frames_item.dir,
            local_path=buffer,
            overwrite=True,
            item_metadata={
                "system": {
                    "shebang": {
                        "dltype": "PCDFrames"
                    }
                },
                "fps": 1
            }
        )
        return frames_item
    finally:
        shutil.rmtree(base_path, ignore_errors=True)


def main():
    dataset_id = "6578a1c89af20782dfb1f4ac"
    filepath = "/frames.json"

    dataset = dl.datasets.get(dataset_id=dataset_id)
    frames_item = dataset.items.get(filepath=filepath)
    print(update_frames_item(frames_item=frames_item))


if __name__ == '__main__':
    main()
