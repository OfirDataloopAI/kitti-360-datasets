import dtlpy as dl
import os
import json
import uuid
import datetime
import shutil
from zipfile import ZipFile
from scipy.spatial.transform import Rotation
import math
from io import BytesIO
import numpy as np
import open3d as o3d
import pathlib

from dtlpylidar.parsers.base_parser import LidarFileMappingParser
import custom_converter_utils as cc_utils


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
        theta_z = 180
        rotation_fix = FixTransformation.rotate_system(theta_z=theta_z, radians=False)
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
    def __init__(self):
        self.attributes_id_mapping_dict = None

        # Data params
        self.number_of_frames = 10
        self.camera_list = ["image_00", "image_01"]
        self.data_3d_path = os.path.join("data_3d_test_slam", "test_0", "2013_05_28_drive_0008_sync")
        self.data_2d_path = os.path.join("data_2d_test_slam", "test_0", "2013_05_28_drive_0008_sync")

        # Calibration params
        self.calibration_path = "calibration"
        self.lidar_extrinsic_filename = "calib_sick_to_velo.txt"
        self.cameras_intrinsic_filename = "perspective.txt"
        self.cameras_extrinsic_filename = "calib_cam_to_pose.txt"
        super().__init__()

    # def attributes_id_mapping(self, dataset):
    #     recipe = dataset.recipes.list()[0]
    #     attributes_mapping = {}
    #
    #     instructions = recipe.metadata.get('system', dict()).get('script', dict()).get('entryPoints', dict()).get(
    #         'annotation:context:set', dict()).get('_instructions', list())
    #
    #     for instruction in instructions:
    #         instructions2 = instruction.get('body', dict()).get('block', dict()).get('_instructions', list())
    #         for instruction2 in instructions2:
    #             title = instruction2.get('title', None)
    #             key = instruction2.get('body', dict()).get('key', None)
    #             attributes_mapping[title] = key
    #     self.attributes_id_mapping_dict = attributes_mapping

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

    def data_pre_processing(self, data_path: str):
        data_3d_path = os.path.join(data_path, self.data_3d_path)

        bin_filepaths = pathlib.Path(data_3d_path).rglob("*.bin")
        bin_filepaths = sorted(list(bin_filepaths))[0:self.number_of_frames]
        for bin_filepath in bin_filepaths:
            input_bin_filepath = str(bin_filepath)
            pcd = cc_utils.convert_kitti_bin_to_pcd(binFilePath=input_bin_filepath)
            output_pcd_filename = input_bin_filepath.replace(".bin", ".pcd")
            o3d.io.write_point_cloud(filename=output_pcd_filename, pointcloud=pcd, write_ascii=True)

    def upload_pcds_and_images(self, data_path: str, dataset: dl.Dataset):
        data_2d_path = os.path.join(data_path, self.data_2d_path)
        data_3d_path = os.path.join(data_path, self.data_3d_path)

        # Get pcd filepaths
        pcd_filepaths = pathlib.Path(data_3d_path).rglob("*.pcd")
        pcd_filepaths = sorted(list(pcd_filepaths))

        # Loop through frames
        for lidar_frame, pcd_filepath in enumerate(pcd_filepaths):
            pcd_filepath = str(pcd_filepath)
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
            pcd_name = os.path.basename(pcd_filepath)

            # Loop through images
            for idx, camera_name in enumerate(self.camera_list):
                image_name = pcd_name.replace(".pcd", ".png")
                image_filepath = os.path.join(data_2d_path, camera_name)
                image_filepath = list(pathlib.Path(image_filepath).rglob(f"*{image_name}"))[0]
                image_filepath = str(image_filepath)

                # Get sensor
                ext = os.path.splitext(p=image_filepath)[1]
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
        output_frames = dict()

        data_2d_path = os.path.join(data_path, self.data_2d_path)
        data_3d_path = os.path.join(data_path, self.data_3d_path)

        # Get pcd filepaths
        pcd_filepaths = pathlib.Path(data_3d_path).rglob("*.pcd")
        pcd_filepaths = sorted(list(pcd_filepaths))

        # Get lidar timestamps
        lidar_timestamp_filepath = list(pathlib.Path(data_3d_path).rglob("*.txt"))[0]
        with open(lidar_timestamp_filepath, "r") as f:
            lidar_timestamps = f.readlines()
            lidar_timestamps = [timestamp.strip() for timestamp in lidar_timestamps]

        images_timestamps = dict()
        for camera_name in self.camera_list:
            camera_timestamp_filepath = os.path.join(data_2d_path, camera_name)
            camera_timestamp_filepath = list(pathlib.Path(camera_timestamp_filepath).rglob(f"*timestamps.txt"))[0]
            with open(camera_timestamp_filepath, "r") as f:
                camera_timestamps = f.readlines()
                camera_timestamps = [timestamp.strip() for timestamp in camera_timestamps]
                images_timestamps[camera_name] = camera_timestamps

        # Get calibration data
        calibration_path = os.path.join(data_path, self.calibration_path)
        lidar_extrinsic_filepath = os.path.join(calibration_path, self.lidar_extrinsic_filename)
        cameras_intrinsic_filepath = os.path.join(calibration_path, self.cameras_intrinsic_filename)
        cameras_extrinsic_filepath = os.path.join(calibration_path, self.cameras_extrinsic_filename)

        lidar_extrinsic = cc_utils.loadCalibrationRigid(filename=lidar_extrinsic_filepath)
        cameras_extrinsic = cc_utils.loadCalibrationCameraToPose(filename=cameras_extrinsic_filepath)
        cameras_intrinsic = cc_utils.loadPerspectiveIntrinsic(filename=cameras_intrinsic_filepath)

        # Loop through frames
        for lidar_frame, pcd_filepath in enumerate(pcd_filepaths):
            pcd_name = os.path.basename(pcd_filepath)

            # Timestamp format: 'yyyy-mm-dd hh:mm:ss.sssssssss'
            lidar_timestamp = lidar_timestamps[lidar_frame]
            lidar_timestamp_datetime = datetime.datetime.strptime(lidar_timestamp[:26], "%Y-%m-%d %H:%M:%S.%f")
            lidar_timestamp_unix = float(lidar_timestamp_datetime.timestamp())

            # Lidar Extrinsic
            position, quaternion = cc_utils.get_position_and_quaternion_from_matrix(matrix=lidar_extrinsic)

            # Output frame dict from `Metadata`
            output_frame_dict = {
                "metadata": {
                    "timestamp": lidar_timestamp,
                    "lidar_name": pcd_name
                },
                "path": f"lidar/{lidar_frame}.pcd",
                "timestamp": lidar_timestamp_unix,
                "position": {
                    "x": position[0],
                    "y": position[1],
                    "z": position[2]
                },
                "heading": {
                    "x": quaternion[0],
                    "y": quaternion[1],
                    "z": quaternion[2],
                    "w": quaternion[3]
                },
                "images": dict()
            }

            # Loop through images
            for idx, camera_name in enumerate(self.camera_list):
                image_name = pcd_name.replace(".pcd", ".png")
                image_filepath = os.path.join(data_2d_path, camera_name)
                image_filepath = list(pathlib.Path(image_filepath).rglob(f"*{image_name}"))[0]

                # Timestamp format: 'yyyy-mm-dd hh:mm:ss.sssssssss'
                image_timestamp = images_timestamps[camera_name][idx]
                image_timestamp_datetime = datetime.datetime.strptime(image_timestamp[:26], "%Y-%m-%d %H:%M:%S.%f")
                image_timestamp_unix = float(image_timestamp_datetime.timestamp())

                # Image Extrinsic and Intrinsic
                camera_extrinsic = cameras_extrinsic[camera_name]
                intrinsic_format = f"P_rect_{camera_name.split('_')[-1]}"
                camera_intrinsic = cameras_intrinsic[intrinsic_format]

                position, quaternion = cc_utils.get_position_and_quaternion_from_matrix(matrix=camera_extrinsic)
                translation, rotation = FixTransformation.fix_camera_transformation(
                    quaternion=quaternion,
                    position=position
                )

                # Output image dict
                ext = os.path.splitext(p=image_filepath)[1]
                image_dict = {
                    "metadata": {
                        "timestamp": image_timestamp,
                        "image_name": image_name
                    },
                    "image_path": f"frames/{lidar_frame}/{idx}{ext}",
                    "timestamp": image_timestamp_unix,
                    "intrinsics": {
                        "fx": camera_intrinsic[0][0],
                        "cx": camera_intrinsic[0][2],
                        "fy": camera_intrinsic[1][1],
                        "cy": camera_intrinsic[1][2]
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
                        "k1": 0,
                        "k2": 0,
                        "k3": 0,
                        "p1": 0,
                        "p2": 0
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
            self.data_pre_processing(data_path=data_path)
            self.upload_pcds_and_images(data_path=data_path, dataset=lidar_dataset)
            mapping_item = self.create_mapping_json(data_path=data_path, dataset=lidar_dataset)
            frames_item = self.parse_data(mapping_item=mapping_item)
            # self.upload_pre_annotation_lidar(frames_item=frames_item, data_path=data_path)
            # self.upload_pre_annotation_images(frames_item=frames_item, data_path=data_path)
        finally:
            shutil.rmtree(path=data_path, ignore_errors=True)

        return frames_item


def main():
    cp = LidarCustomParser()

    dl.setenv('prod')
    data_path = "./KITTI-360/test_data"
    dataset = dl.datasets.get(dataset_id="66602cc51fb4fc872de5cfca")

    # cp.data_pre_processing(data_path=data_path)
    # cp.upload_pcds_and_images(data_path=data_path, dataset=dataset)
    mapping_item = cp.create_mapping_json(data_path=data_path, dataset=dataset)
    # mapping_item = dataset.items.get(item_id="65f32fa45c63d275df8dc81d")
    frames_item = cp.parse_data(mapping_item=mapping_item)

    # frames_item = dataset.items.get(filepath="/frames.json")
    # cp.upload_pre_annotation_lidar(frames_item=frames_item, data_path=data_path)
    # cp.upload_pre_annotation_images(frames_item=frames_item, data_path=data_path)


if __name__ == "__main__":
    main()
