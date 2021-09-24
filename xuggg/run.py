import argparse
import os
import torch.nn as nn
from model import Model
from torch.utils.data import DataLoader
from dataset.dataset import Dataset


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        type=str,
                        default="../Dataset/CelebA")
    parser.add_argument('--mask_path', type=str, default="../Dataset/Mask")
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--result_save_path', type=str, default='results')
    parser.add_argument('--target_size', type=int, default=256)
    parser.add_argument('--mask_mode', type=int, default=2)
    parser.add_argument('--num_iters', type=int, default=1000000)
    parser.add_argument('--model_path',
                        type=str,
                        default='checkpoints_paris/g_700000.pth')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--n_threads', type=int, default=2)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--gpu_id', type=str, default='1')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "2, 3"
    model = Model()
    # args.finetune = True
    if args.test:
        model.initialize_model(args.model_path, train=False)
        model.cuda()
        model = nn.DataParallel(model)
        dataloader = DataLoader(
            Dataset(args.data_path,
                    args.mask_path,
                    args.mask_mode,
                    args.target_size,
                    mask_reverse=True,
                    training=False))
        model.module.test(dataloader, args.result_save_path)
    else:
        model.initialize_model(args.model_path, train=True)
        model.cuda()
        model = nn.DataParallel(model)
        dataloader = DataLoader(Dataset(args.data_path,
                                        args.mask_path,
                                        args.mask_mode,
                                        args.target_size,
                                        mask_reverse=True),
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.n_threads,
                                drop_last=True)
        model.module.train(dataloader, args.model_save_path, args.finetune,
                           args.num_iters)


if __name__ == '__main__':
    run()
# python run.py --data_path ../Dataset/CelebA256 --mask_path ../Dataset/Masks
# python run.py --data_path ../Dataset/CelebA --mask_path ../Dataset/Masks --gpu_id 3 --model_path checkpoints/g_200.pth --test
