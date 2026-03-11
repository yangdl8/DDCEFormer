from tqdm import tqdm
import torch.utils.data
from common.utils import *
from common.loss import *
from common.load_data_hm36 import Fusion
from common.h36m_dataset import Human36mDataset
from common.arguments import parse_args
from model.DDCEFormer import Model

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


if __name__ == '__main__':
    dataset_path = args.root_path + 'data_3d_' + args.dataset + '.npz'
    dataset = Human36mDataset(dataset_path, args)
    actions = define_actions(args.actions)

    test_data = Fusion(args, dataset, args.root_path, train=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                            shuffle=False, num_workers=int(args.workers), pin_memory=True)

    model = Model(args).cuda()
    if args.previous_dir != '':
        Load_model(args, model)

    with torch.no_grad():
        model.eval()
        action_error = define_error_list(actions)
        joints_left = [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]

        for i, data in enumerate(tqdm(test_dataloader, dynamic_ncols=True)):
            batch_cam, gt_3D, input_2D, input_2D_GT, action, subject, cam_ind = data
            input_2D, input_2D_GT, gt_3D, batch_cam = input_2D.cuda(), input_2D_GT.cuda(), gt_3D.cuda(), batch_cam.cuda()

            output_3D_non_flip = model(input_2D[:, 0])
            output_3D_flip = model(input_2D[:, 1])
            output_3D_flip[:, :, :, 0] *= -1
            output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]
            output_3D = (output_3D_non_flip + output_3D_flip) / 2

            out_target = gt_3D.clone()
            output_3D[:, :, args.root_joint] = 0
            out_target[:, :, args.root_joint] = 0

            action_error = test_calculation(output_3D, out_target, action, action_error, args.dataset, subject)

        print_error(args.dataset, action_error, 0)
