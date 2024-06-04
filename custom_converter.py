import dtlpy as dl
import os
import json
import uuid
import datetime
import shutil
from zipfile import ZipFile
from scipy.spatial.transform import Rotation
import math
import numpy as np
from io import BytesIO

from dtlpylidar.parsers.base_parser import LidarFileMappingParser


class FixTransformation:
    @staticmethod
    def rotate_system(theta_x=None, theta_y=None, theta_z=None, radians: bool = True):
        if radians is False:
            theta_x = math.radians(theta_x) if theta_x else None
            theta_y = math.radians(theta_y) if theta_y else None
            theta_z = math.radians(theta_z) if theta_z else None

        rotation = np.identity(4)
        if theta_x is not None:
            rotation_x = np.array([
                [1, 0, 0, 0],
                [0, math.cos(theta_x), -math.sin(theta_x), 0],
                [0, math.sin(theta_x), math.cos(theta_x), 0],
                [0, 0, 0, 1]
            ])
            rotation = rotation @ rotation_x
        if theta_y is not None:
            rotation_y = np.array([
                [math.cos(theta_y), 0, math.sin(theta_y), 0],
                [0, 1, 0, 0],
                [-math.sin(theta_y), 0, math.cos(theta_y), 0],
                [0, 0, 0, 1]
            ])
            rotation = rotation @ rotation_y
        if theta_z is not None:
            rotation_z = np.array([
                [math.cos(theta_z), -math.sin(theta_z), 0, 0],
                [math.sin(theta_z), math.cos(theta_z), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            rotation = rotation @ rotation_z
        rotation[np.abs(rotation) < 1e-5] = 0
        return rotation

    @staticmethod
    def fix_camera_transformation(quaternion: np.ndarray, position: np.ndarray):
        # Rotation
        rotation_matrix = np.identity(4)
        rotation_matrix[0:3, 0:3] = Rotation.from_quat(quaternion).as_matrix()

        # Apply Rotation fix
        theta_y = 90
        theta_z = -90
        rotation_fix = FixTransformation.rotate_system(theta_y=theta_y, theta_z=theta_z, radians=False)
        rotation_matrix = rotation_matrix @ rotation_fix

        # Translation
        translation_matrix = np.identity(4)
        translation_matrix[0:3, 3] = position.tolist()

        # Extrinsic Matrix
        extrinsic_matrix = translation_matrix @ rotation_matrix
        translation_array = extrinsic_matrix[0:3, 3]
        translation = {"x": translation_array[0], "y": translation_array[1], "z": translation_array[2]}
        rotation_array = Rotation.as_quat(Rotation.from_matrix(extrinsic_matrix[0:3, 0:3]))
        # rotation_array = FixTransformation.rotate_coordinates(rotation=quaternion)
        rotation = {"x": rotation_array[0], "y": rotation_array[1], "z": rotation_array[2], "w": rotation_array[3]}
        return translation, rotation


class LidarCustomParser(LidarFileMappingParser):
    def __init__(self,
                 enable_ir_cameras: str,
                 enable_rgb_cameras: str,
                 enable_rgb_highres_cameras: str):
        self.attributes_id_mapping_dict = None
        # Handle Cameras Options
        ir_cameras = ['ir_center', 'ir_left', 'ir_right']
        rgb_cameras = ['rgb_center', 'rgb_left', 'rgb_right']
        rgb_highres_cameras = ['rgb_highres_center', 'rgb_highres_left', 'rgb_highres_right']

        self.camera_list = list()
        if str(enable_ir_cameras) == "true" or str(enable_ir_cameras) == "True":
            self.camera_list += ir_cameras
        if str(enable_rgb_cameras) == "true" or str(enable_rgb_cameras) == "True":
            self.camera_list += rgb_cameras
        if str(enable_rgb_highres_cameras) == "true" or str(enable_rgb_highres_cameras) == "True":
            self.camera_list += rgb_highres_cameras

        super().__init__()

    def attributes_id_mapping(self, dataset):
        recipe = dataset.recipes.list()[0]
        attributes_mapping = {}

        instructions = recipe.metadata.get('system', dict()).get('script', dict()).get('entryPoints', dict()).get(
            'annotation:context:set', dict()).get('_instructions', list())

        for instruction in instructions:
            instructions2 = instruction.get('body', dict()).get('block', dict()).get('_instructions', list())
            for instruction2 in instructions2:
                title = instruction2.get('title', None)
                key = instruction2.get('body', dict()).get('key', None)
                attributes_mapping[title] = key
        self.attributes_id_mapping_dict = attributes_mapping

    @staticmethod
    def extract_zip_file(zip_filepath: str):
        data_path = str(uuid.uuid4())

        try:
            os.makedirs(name=data_path, exist_ok=True)

            with ZipFile(zip_filepath, 'r') as zip_object:
                zip_object.extractall(path=os.path.join(".", data_path))

        except Exception as e:
            shutil.rmtree(path=data_path, ignore_errors=True)
            raise dl.exceptions.BadRequest(
                status_code="400",
                message=f"Failed due to the following error: {e}"
            )
        data_path = os.path.join(os.getcwd(), data_path)
        return data_path

    @staticmethod
    def create_lidar_dataset(item: dl.Item, overwrite: bool):
        dataset_name = item.name

        try:
            dataset = item.project.datasets.create(dataset_name=dataset_name)
        except:
            if overwrite:
                dataset = item.project.datasets.get(dataset_name=dataset_name)
            else:
                dataset_name += f'_{datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}'
                dataset = item.project.datasets.create(dataset_name=dataset_name)

        return dataset

    def upload_pcds_and_images(self, data_path: str, dataset: dl.Dataset):
        scene = None
        dir_items = os.listdir(path=data_path)
        for dir_item in dir_items:
            if ".json" in dir_item:
                try:
                    calibration_json = os.path.join(data_path, dir_item)
                    # scene = raillabel.load(calibration_json)
                    break
                except:
                    continue

        if scene is None:
            dl.exceptions.NotFound("Couldn't find supported json for 'raillabel'")

        # Loop through frames
        frames = scene.frames
        for lidar_frame, (frame_num, frame) in enumerate(frames.items()):
            # Sensor ego pose
            ego_pose = frame.sensors['lidar']
            pcd_filepath = os.path.join(data_path, ego_pose.uri[1:])
            dataset.items.upload(
                local_path=pcd_filepath,
                remote_path=f"/lidar",
                remote_name=f"{lidar_frame}.pcd",
                overwrite=True
            )
            print(
                f"Uploaded to 'lidar' directory, the file: '{pcd_filepath}', "
                f"as: '/lidar/{lidar_frame}.pcd'"
            )

            # Loop through images
            for idx, image in enumerate(self.camera_list):
                # Get sensor
                sensor_reference = frame.sensors[image]
                image_filepath = os.path.join(data_path, sensor_reference.uri[1:])
                ext = os.path.splitext(p=sensor_reference.uri)[1]
                dataset.items.upload(
                    local_path=image_filepath,
                    remote_path=f"/frames/{lidar_frame}",
                    remote_name=f"{idx}{ext}",
                    overwrite=True
                )
                print(
                    f"Uploaded to 'frames' directory, the file: '{image_filepath}', "
                    f"as: '/frames/{lidar_frame}/{idx}{ext}'"
                )

    def create_mapping_json(self, data_path: str, dataset: dl.Dataset):
        scene = None
        dir_items = os.listdir(path=data_path)
        for dir_item in dir_items:
            if ".json" in dir_item:
                try:
                    calibration_json = os.path.join(data_path, dir_item)
                    # scene = raillabel.load(calibration_json)
                    break
                except:
                    continue

        if scene is None:
            dl.exceptions.NotFound("Couldn't find supported json for 'raillabel'")

        output_frames = dict()

        # Loop through frames
        frames = scene.frames
        for lidar_frame, (frame_num, frame) in enumerate(frames.items()):
            # Sensor ego pose
            ego_pose = frame.sensors['lidar']

            # Output frame dict from `Metadata`
            output_frame_dict = {
                "metadata": {
                    "frame": int(frame_num),
                    "sensor_uri": ego_pose.uri
                },
                "path": f"lidar/{lidar_frame}.pcd",
                "timestamp": float(frame.timestamp),
                "position": {
                    "x": ego_pose.sensor.extrinsics.pos.x,
                    "y": ego_pose.sensor.extrinsics.pos.y,
                    "z": ego_pose.sensor.extrinsics.pos.z
                },
                "heading": {
                    "x": ego_pose.sensor.extrinsics.quat.x,
                    "y": ego_pose.sensor.extrinsics.quat.y,
                    "z": ego_pose.sensor.extrinsics.quat.z,
                    "w": ego_pose.sensor.extrinsics.quat.w
                },
                "images": dict()
            }

            # Loop through images
            for idx, image in enumerate(self.camera_list):
                # Get sensor
                sensor_reference = frame.sensors[image]

                # Get extrinsics
                extrinsics = sensor_reference.sensor.extrinsics
                quaternion = np.array([extrinsics.quat.x, extrinsics.quat.y, extrinsics.quat.z, extrinsics.quat.w])
                position = np.array([extrinsics.pos.x, extrinsics.pos.y, extrinsics.pos.z])

                # Apply camera transformation fix
                translation, rotation = FixTransformation.fix_camera_transformation(
                    quaternion=quaternion,
                    position=position
                )

                # Output image dict
                ext = os.path.splitext(p=sensor_reference.uri)[1]
                image_dict = {
                    "metadata": {
                        "frame": int(frame_num),
                        "image_uri": sensor_reference.uri,
                    },
                    "image_path": f"frames/{lidar_frame}/{idx}{ext}",
                    "timestamp": float(sensor_reference.timestamp),
                    "intrinsics": {
                        "fx": sensor_reference.sensor.intrinsics.camera_matrix[0],
                        "fy": sensor_reference.sensor.intrinsics.camera_matrix[5],
                        "cx": sensor_reference.sensor.intrinsics.camera_matrix[2],
                        "cy": sensor_reference.sensor.intrinsics.camera_matrix[6],
                    },
                    "extrinsics": {
                        "translation": {
                            "x": translation["x"],
                            "y": translation["y"],
                            "z": translation["z"]
                        },
                        "rotation": {
                            "x": rotation["x"],
                            "y": rotation["y"],
                            "z": rotation["z"],
                            "w": rotation["w"]
                        },
                    },
                    "distortion": {
                        "k1": sensor_reference.sensor.intrinsics.distortion[0],
                        "k2": sensor_reference.sensor.intrinsics.distortion[1],
                        "k3": sensor_reference.sensor.intrinsics.distortion[2],
                        "p1": sensor_reference.sensor.intrinsics.distortion[3],
                        "p2": sensor_reference.sensor.intrinsics.distortion[4]
                    }
                }
                output_frame_dict['images'][str(idx)] = image_dict

            output_frames[str(lidar_frame)] = output_frame_dict

        mapping_data = {"frames": output_frames}
        mapping_filepath = os.path.join(data_path, "mapping.json")
        with open(mapping_filepath, "w") as f:
            json.dump(obj=mapping_data, fp=f, indent=4)

        mapping_item = dataset.items.upload(local_path=mapping_filepath, overwrite=True)
        return mapping_item

    def custom_parse_data(self, zip_filepath: str, lidar_dataset: dl.Dataset):
        data_path = self.extract_zip_file(zip_filepath=zip_filepath)

        try:
            self.upload_pcds_and_images(data_path=data_path, dataset=lidar_dataset)
            mapping_item = self.create_mapping_json(data_path=data_path, dataset=lidar_dataset)
            frames_item = self.parse_data(mapping_item=mapping_item)
            self.upload_pre_annotation_lidar(frames_item=frames_item, data_path=data_path)
            self.upload_pre_annotation_images(frames_item=frames_item, data_path=data_path)
        finally:
            shutil.rmtree(path=data_path, ignore_errors=True)

        return frames_item


def main():
    cp = LidarCustomParser(
        enable_ir_cameras="false",
        enable_rgb_cameras="false",
        enable_rgb_highres_cameras="true"
    )

    data_path = "./data"
    dataset = dl.datasets.get(dataset_id="66099e6289c8593e33498ce1")

    # cp.upload_pcds_and_images(data_path=data_path, dataset=dataset)
    mapping_item = cp.create_mapping_json(data_path=data_path, dataset=dataset)
    frames_item = cp.parse_data(mapping_item=mapping_item)
    # mapping_item = dataset.items.get(item_id="65f32fa45c63d275df8dc81d")

    # frames_item = dataset.items.get(filepath="/frames.json")
    cp.upload_pre_annotation_lidar(frames_item=frames_item, data_path=data_path)
    # cp.upload_pre_annotation_images(frames_item=frames_item, data_path=data_path)

    # cp.upload_radar_with_annotations(dataset=dataset, data_path=data_path)


if __name__ == "__main__":
    main()
