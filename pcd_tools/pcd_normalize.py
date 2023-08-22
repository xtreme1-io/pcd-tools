from pathlib import Path
from pcd_tools.load_pcd import PointCloud


def normalize_pcd(pcd_file: str, out_pcd_file: str, extra_fields: list):
    pc = PointCloud(pcd_file)
    npc = pc.normalized_pc(extra_fields)
    pc.save_pcd(npc, out_pcd_file)


def main():
    import argparse
    from pcd_tools import Timing

    parser = argparse.ArgumentParser()
    parser.add_argument('pcd_file', type=str, help='pcd file path')
    parser.add_argument('out_pcd_file', type=str, default=None, nargs='?', help='output pcd file path')
    args = parser.parse_args()

    pcd_file = args.pcd_file
    out_pcd_file = args.out_pcd_file 
    if out_pcd_file is None:
        src_path = Path(pcd_file)
        out_pcd_file = str(src_path.with_stem(src_path.stem + "-binary"))

    with Timing():
        normalize_pcd(args.pcd_file, out_pcd_file, extra_fields=[])


if __name__ == '__main__':
    main()
