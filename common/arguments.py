import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--k_ratio', type=float, default=0.75)
    parser.add_argument('--g_ratio', type=float, default=1.5)
    parser.add_argument('--channel', default=512, type=int)
    parser.add_argument('--d_hid', default=1024, type=int)
    parser.add_argument('--frames', type=int, default=243)
    parser.add_argument('--pad', type=int, default=121)
    parser.add_argument('--dataset', type=str, default='h36m')
    parser.add_argument('--keypoints', default='cpn_ft_h36m_dbb', type=str)
    parser.add_argument('--data_augmentation', type=int, default=1)
    parser.add_argument('--reverse_augmentation', type=bool, default=False)
    parser.add_argument('--test_augmentation', type=bool, default=True)
    parser.add_argument('--crop_uv', type=int, default=0)
    parser.add_argument('--root_path', type=str, default='./dataset/')
    parser.add_argument('--actions', default='*', type=str)
    parser.add_argument('--downsample', default=1, type=int)
    parser.add_argument('--subset', default=1, type=float)
    parser.add_argument('--stride', default=243, type=int)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--previous_dir', type=str, default='pretrained')
    parser.add_argument('--n_joints', type=int, default=17)
    parser.add_argument('--out_all', type=int, default=1)

    args = parser.parse_args()

    args.pad = (args.frames-1) // 2

    args.root_joint = 0
    args.subjects_train = 'S1,S5,S6,S7,S8'
    args.subjects_test = 'S9,S11'

    return args
