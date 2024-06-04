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
    bin_filename = r"C:\Users\Ofir\PycharmProjects\kitti-360-datasets\KITTI-360\data_3d_raw\2013_05_28_drive_0000_sync\velodyne_points\data\0000000000.bin"
    pcd_filename = "./output.pcd"

    pcd = convert_kitti_bin_to_pcd(binFilePath=bin_filename)
    o3d.io.write_point_cloud(filename=pcd_filename, pointcloud=pcd, write_ascii=True)


if __name__ == '__main__':
    main()
