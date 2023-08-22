from PIL import Image
import numpy as np
from pathlib import Path
from pcd_tools.load_pcd import PointCloud


class PC2Image:
    def __init__(self, colors=(0x6055C6, 0x378CDF, 0x1CC7C1, 0x33EC83, 0x7FF55B), 
        z_range=None, resolution=(1024, 1024), num_std=3):
        """
        :param colors: palette
        :param z_range: (z_min, z_max)
        :param resolution: (W, H)
        :param num_std: number of standard deviations
        """
        assert z_range is None or (isinstance(z_range, (tuple, list)) and len(z_range) == 2)

        colors_np = np.zeros((len(colors), 3), dtype=np.uint8)
        colors_ = np.asarray(colors)
        colors_np[:, 0] = (colors_ >> 16) & 0xFF
        colors_np[:, 1] = (colors_ >> 8) & 0xFF
        colors_np[:, 2] = colors_ & 0xFF
        self.colors = colors_np
        self.z_range = None if z_range is None else np.asarray(z_range)

        assert isinstance(resolution, (tuple, list)) and len(resolution) == 2
        self.resolution = np.array(resolution)
        self.num_std = num_std

    def bev_from_pc(self, pc: np.ndarray):
        points = pc[:, :3] if pc.shape[1] > 3 else pc
        xy = points[:, :2]

        # calculate size
        _mean = xy.mean(axis=0)
        _std = xy.std(axis=0)
        _min = np.maximum(_mean - _std * self.num_std, xy.min(axis=0))
        _max = np.minimum(_mean + _std * self.num_std, xy.max(axis=0))
        _size = _max - _min

        c_size = (_size / self.resolution).max()
        half_size = (c_size * self.resolution) / 2
        _min = _mean - half_size
        _max = _mean + half_size

        # sort by z descending
        indices = points[:, 2].argsort(axis=0)
        points = points[indices]

        # projection
        uvs = ((points[:, :2] - _min) / c_size).round().astype(int)

        # filter points
        mask = ((uvs > [0, 0]) & (uvs < self.resolution)).all(axis=1)
        uvs = uvs[mask]

        # colors
        z_range = self.z_range
        if z_range is None:
            z = pc[:, 2]
            # z_mean = z.mean()
            # z_std = z.std()
            # z_range = [
            #     np.maximum(z_mean - z_std*3, z.min()), 
            #     np.minimum(z_mean + z_std*3, z.max())]
            z_range = [z.min(), z.max()]
        z_splits = np.linspace(*z_range, len(self.colors)+1)[1:-1]
        color_indices = np.digitize(points[:, 2][mask], z_splits)
        W, H = self.resolution
        _img = np.zeros((H, W, 3), dtype=np.uint8)
        _img[-uvs[:, 1], uvs[:, 0]] = self.colors[color_indices]

        left, bottom = _min.round(3)
        right, top = _max.round(3)
        return Image.fromarray(_img), (left, top, right, bottom)

    def pcd2png(self, pcd_file, png_file):
        pc = PointCloud(pcd_file).numpy(fields='xyz')
        image, _ = self.bev_from_pc(pc)

        pcd_path = Path(pcd_file)
        png_path = pcd_path.with_suffix('.png') if png_file is None else Path(png_file)
        png_path.parent.mkdir(exist_ok=True, parents=True)
        image.save(png_path)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('pcd_file', type=str, help='pcd file path')
    parser.add_argument('png_file', type=str, help='png file path', default=None, nargs='?')

    args = parser.parse_args()
    PC2Image().pcd2png(args.pcd_file, args.png_file)


if __name__ == '__main__':
    main()
