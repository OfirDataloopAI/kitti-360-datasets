import numpy as np
import struct
import open3d as o3d


def convert_kitti_bin_to_pcd(binFilePath):
    size_float = 4
    list_pcd = []
    with open(binFilePath, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd)
    return pcd


def main():
    bin_filenames = [
        r".\KITTI-360\test_data\data_3d_test_slam\test_0\2013_05_28_drive_0008_sync\velodyne_points\data\0000001482.bin",
        r".\KITTI-360\test_data\data_3d_test_slam\test_0\2013_05_28_drive_0008_sync\velodyne_points\data\0000001483.bin",
        r".\KITTI-360\test_data\data_3d_test_slam\test_0\2013_05_28_drive_0008_sync\velodyne_points\data\0000001484.bin",
        r".\KITTI-360\test_data\data_3d_test_slam\test_0\2013_05_28_drive_0008_sync\velodyne_points\data\0000001485.bin",
        r".\KITTI-360\test_data\data_3d_test_slam\test_0\2013_05_28_drive_0008_sync\velodyne_points\data\0000001486.bin",
        r".\KITTI-360\test_data\data_3d_test_slam\test_0\2013_05_28_drive_0008_sync\velodyne_points\data\0000001487.bin",
        r".\KITTI-360\test_data\data_3d_test_slam\test_0\2013_05_28_drive_0008_sync\velodyne_points\data\0000001488.bin",
        r".\KITTI-360\test_data\data_3d_test_slam\test_0\2013_05_28_drive_0008_sync\velodyne_points\data\0000001489.bin",
        r".\KITTI-360\test_data\data_3d_test_slam\test_0\2013_05_28_drive_0008_sync\velodyne_points\data\0000001490.bin",
        r".\KITTI-360\test_data\data_3d_test_slam\test_0\2013_05_28_drive_0008_sync\velodyne_points\data\0000001491.bin"
    ]

    for bin_filename in bin_filenames:
        pcd_filename = bin_filename.replace(".bin", ".pcd").replace(r"\data\0", r"\pcd\0")
        pcd = convert_kitti_bin_to_pcd(binFilePath=bin_filename)
        o3d.io.write_point_cloud(filename=pcd_filename, pointcloud=pcd, write_ascii=True)


if __name__ == '__main__':
    main()
