import random
import math
import logging
from tqdm import tqdm
import torch.utils.data
import torch.optim as optim
from common.utils import *
from common.loss import *
from common.load_data_hm36 import Fusion
from common.h36m_dataset import Human36mDataset
from common.arguments import parse_args
from model.DDCEFormer import Model

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def train(dataloader, model, optimizer, epoch):
    model.train()
    loss_all = {'loss': AccumLoss(), 'loss_mpjpe': AccumLoss(), 'n_mpjpe': AccumLoss(), \
    	'velocity': AccumLoss(), 'n_mpjpe_k': AccumLoss(), 'velocity_k': AccumLoss()}

    for i, data in enumerate(tqdm(dataloader)):
        batch_cam, gt_3D, input_2D, input_2D_GT, action, subject, cam_ind = data
        input_2D, input_2D_GT, gt_3D, batch_cam = input_2D.cuda(), input_2D_GT.cuda(), gt_3D.cuda(), batch_cam.cuda()

        output_3D = model(input_2D)

        out_target = gt_3D.clone()
        out_target[:, :, args.root_joint] = 0

        # Loss
        args_lambda_scale = 0.5
        args_lambda_3d_velocity = 20.0
        args_lambda_lv = 0.0
        args_lambda_lg = 0.0
        args_lambda_a = 0.0
        args_lambda_av = 0.0

        loss_3d_pos = loss_mpjpe(output_3D, out_target)
        loss_3d_scale = n_mpjpe(output_3D, out_target)
        loss_3d_velocity = loss_velocity(output_3D, out_target)
        loss_3d_scale_k = args_lambda_scale * loss_3d_scale
        loss_3d_velocity_k = args_lambda_3d_velocity * loss_3d_velocity

        loss_lv = loss_limb_var(output_3D)
        loss_lg = loss_limb_gt(output_3D, out_target)
        loss_a = loss_angle(output_3D, out_target)
        loss_av = loss_angle_velocity(output_3D, out_target)
        loss = loss_3d_pos + \
                args_lambda_scale * loss_3d_scale + \
                args_lambda_3d_velocity * loss_3d_velocity + \
                args_lambda_lv * loss_lv + \
                args_lambda_lg * loss_lg + \
                args_lambda_a * loss_a + \
                args_lambda_av * loss_av

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        N = input_2D.shape[0]
        loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)
        loss_all['loss_mpjpe'].update(loss_3d_pos.detach().cpu().numpy() * N, N)
        loss_all['n_mpjpe'].update(loss_3d_scale.detach().cpu().numpy() * N, N)
        loss_all['velocity'].update(loss_3d_velocity.detach().cpu().numpy() * N, N)
        loss_all['n_mpjpe_k'].update(loss_3d_scale_k.detach().cpu().numpy() * N, N)
        loss_all['velocity_k'].update(loss_3d_velocity_k.detach().cpu().numpy() * N, N)
        
    return loss_all['loss'].avg, loss_all['loss_mpjpe'].avg, loss_all['n_mpjpe'].avg, loss_all['velocity'].avg, loss_all['n_mpjpe_k'].avg, loss_all['velocity_k'].avg


def test(actions, dataloader, model):
    model.eval()

    action_error = define_error_list(actions)

    joints_left = [4, 5, 6, 11, 12, 13] 
    joints_right = [1, 2, 3, 14, 15, 16]

    for i, data in enumerate(tqdm(dataloader, dynamic_ncols=True)):
        batch_cam, gt_3D, input_2D, input_2D_GT, action, subject, cam_ind = data
        input_2D, input_2D_GT, gt_3D, batch_cam = input_2D.cuda(), input_2D_GT.cuda(), gt_3D.cuda(), batch_cam.cuda()

        output_3D_non_flip = model(input_2D[:, 0])
        output_3D_flip = model(input_2D[:, 1])

        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :] 

        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        out_target = gt_3D.clone()
        if args.stride == 1:
            out_target = out_target[:, args.pad].unsqueeze(1)
            output_3D = output_3D[:, args.pad].unsqueeze(1)

        output_3D[:, :, args.root_joint] = 0
        out_target[:, :, args.root_joint] = 0

        action_error = test_calculation(output_3D, out_target, action, action_error, args.dataset, subject)

    p1, p2 = print_error(args.dataset, action_error, args.train)

    return p1, p2

if __name__ == '__main__':
    seed = 1126

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    dataset_path = args.root_path + 'data_3d_' + args.dataset + '.npz'
    dataset = Human36mDataset(dataset_path, args)
    actions = define_actions(args.actions)

    if args.train:
        train_data = Fusion(args, dataset, args.root_path, train=True)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                            shuffle=True, num_workers=int(args.workers), pin_memory=True)
    test_data = Fusion(args, dataset, args.root_path, train=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                            shuffle=False, num_workers=int(args.workers), pin_memory=True)

    model = Model(args).cuda()

    if args.previous_dir != '':
        Load_model(args, model)

    lr = args.lr
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    best_epoch1 = 0
    best_epoch2 = 0
    previous_best1 = math.inf
    previous_best2 = math.inf
    previous_name1 = ''
    previous_name2 = ''
    loss_epochs = []
    mpjpes = []

    for epoch in range(1, args.nepoch + 1):
        if args.train: 
            loss, loss_m, loss_n, loss_v, loss_nk, loss_vk = train(train_dataloader, model, optimizer, epoch)

            loss_epochs.append(loss * 1000)

        with torch.no_grad():
            p1, p2 = test(actions, test_dataloader, model)
            mpjpes.append(p1)

        if args.train and p1 < previous_best1:
            best_epoch1 = epoch
            previous_name1 = save_model(args.checkpoint, previous_name1, epoch, p1, model, 'model_p1')
            previous_best1 = p1
        if args.train and p2 < previous_best2:
            best_epoch2 = epoch
            previous_name2 = save_model(args.checkpoint, previous_name2, epoch, p2, model, 'model_p2')
            previous_best2 = p2

        if args.train:
            logging.info('epoch: %d, lr: %.6f, l: %.4f, lm: %.4f, ln: %.4f, lv: %.4f, lnk: %.4f, lvk: %.4f, p1: %.2f, p2: %.2f, %d: %.2f, %d: %.2f' % (epoch, lr, loss, loss_m, loss_n, loss_v, loss_nk, loss_vk, p1, p2, best_epoch1, previous_best1, best_epoch2, previous_best2))
            print('%d, lr: %.6f, l: %.4f, p1: %.2f, p2: %.2f, %d: %.2f, %d: %.2f' % (epoch, lr, loss, p1, p2, best_epoch1, previous_best1, best_epoch2, previous_best2))
        
            if epoch % args.lr_decay_epoch == 0:
                lr *= args.lr_decay_large
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_decay_large
            else:
                lr *= args.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_decay 
        else:
            print('p1: %.2f, p2: %.2f' % (p1, p2))
            break

