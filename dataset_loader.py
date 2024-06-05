import dtlpy as dl
import requests
import os
import logging
import json

import custom_converter as lidar

logger = logging.getLogger(name='osdar-dataset')


class DatasetLidarKITTI(dl.BaseServiceRunner):
    def __init__(self):
        dl.use_attributes_2(state=True)

        self.dataset_url = "https://download.data.fid-move.de/dzsf/osdar23/1_calibration_1.2.zip"
        self.zip_filename = "data.zip"

    def _download_zip(self):
        zip_filepath = os.path.join(os.getcwd(), self.zip_filename)
        with requests.get(self.dataset_url, stream=True) as r:
            r.raise_for_status()
            with open(zip_filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info(msg=f"File downloaded to: {zip_filepath}")
        return zip_filepath

    def upload_dataset(self, dataset: dl.Dataset, source: str):
        # self._import_recipe_ontology(dataset=dataset)
        if self.zip_filename not in os.listdir(path=os.getcwd()):
            zip_filepath = self._download_zip()
        else:
            zip_filepath = os.path.join(os.getcwd(), self.zip_filename)

        item: dl.Item
        lidar_parser = lidar.LidarCustomParser()
        frames_item = lidar_parser.custom_parse_data(zip_filepath=zip_filepath, lidar_dataset=dataset)
        return frames_item


def test_download():
    sr = DatasetLidarKITTI()
    sr._download_zip()


# def test_import_recipe_ontology():
#     dataset_id = "66325a24241a71f884f78431"
#
#     dataset = dl.datasets.get(dataset_id=dataset_id)
#     sr = DatasetLidarKITTI()
#     sr._import_recipe_ontology(dataset=dataset)


def test_dataset_import():
    dataset_id = "663b93cfd03cf2f75ddeff4f"

    dataset = dl.datasets.get(dataset_id=dataset_id)
    sr = DatasetLidarKITTI()
    sr.upload_dataset(dataset=dataset, source="")


def main():
    # test_download()
    # test_import_recipe_ontology()
    test_dataset_import()


if __name__ == '__main__':
    main()
