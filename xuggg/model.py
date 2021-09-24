import torch
import time
import os
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from utils.io import load_ckpt, save_ckpt
from modules.main import Main, VGG16FeatureExtractor
from torch.utils.tensorboard import SummaryWriter


class Model():
    def __init__(self):
        self.G = None
        self.lossNet = None
        self.iter = None
        self.optm_G = None
        self.device = None
        self.real_A = None  # masked_image
        self.fake_A = None
        self.real_B = None  # gt_image
        self.fake_B = None  # G(masked_image)
        self.fake_C = None
        self.comp_B = None  # G(masked_image) * (1 - mask) + gt * mask
        self.l1_loss_val = 0.0
        self.hole_loss = 0.0
        self.laplace_loss_ = 0.0
        self.loss = 0.0

    def initialize_model(self, path=None, train=True):
        self.G = Main()
        self.optm_G = optim.Adam(self.G.parameters(), lr=2e-4)
        if train:
            self.lossNet = VGG16FeatureExtractor()

        try:
            start_iter = load_ckpt(path, [('generator', self.G)],
                                   [('optimizer_G', self.optm_G)])
            if train:
                self.optm_G = optim.Adam(self.G.parameters(), lr=2e-4)
                print('Model Initialized, iter:', start_iter)
                self.iter = start_iter
        except Exception:
            print('No trained model, from Start')
            self.iter = 0

    def cuda(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.G.cuda()
            print('Model moved to cuda')
            if self.lossNet is not None:
                self.lossNet.cuda()
        else:
            self.device = torch.device('cpu')

    def train(self, train_loader, save_path, finetune=False, iters=4000):
        self.G.train(finetune=finetune)
        total = sum([param.nelement() for param in self.G.parameters()])
        print(f'Numbers of params: {total / 1e6:.2f}M')
        if finetune:
            self.optm_G = optim.Adam(filter(lambda p: p.requires_grad,
                                            self.G.parameters()),
                                     lr=6e-6)

        print(f'Starting training from iteration:{self.iter}')
        s_time = time.time()
        while self.iter < iters:
            for items in train_loader:
                gt_images, masks = self.__cuda__(*items)
                masked_images = gt_images * masks
                masks = torch.cat([masks] * 3, dim=1)
                # with torch.no_grad():
                #     # print('mean is : ', torch.mean(masked_images))
                #     self.forward(masked_images, masks, gt_images)
                self.forward(masked_images, masks, gt_images)
                self.update_parameters()
                # loss_G = self.get_g_loss()
                # print('loss: ', loss_G.item())
                self.iter += 1
                iter_p = 50
                if self.iter % iter_p == 0:
                    infolog = open('./iterationlog.txt', 'a+')
                    e_time = time.time()
                    int_time = e_time - s_time
                    info = f'Iteration:{self.iter}, Total_loss: {self.loss / iter_p:.4f}, L1_loss: {self.l1_loss_val / iter_p:.4f}, Hole_loss: {self.hole_loss / iter_p:.4f}, Time_taken: {int_time:.2f}'
                    print(info)
                    infolog.write(info + '\n')
                    infolog.close()

                    writer = SummaryWriter('./lossBoard')
                    writer.add_scalar('trainLoss', self.hole_loss / iter_p,
                                      self.iter)
                    # writer.add_scalar('lapalceLoss',
                    #   self.laplace_loss_ / iter_p, self.iter)
                    writer.add_scalar('Loss', self.loss / iter_p, self.iter)
                    writer.flush()
                    s_time = time.time()
                    self.loss = 0.0
                    self.hole_loss = 0.0
                    self.laplace_loss_ = 0.0
                    self.l1_loss_val = 0.0

                if self.iter % 5000 == 0:
                    if not os.path.exists(f'{save_path}'):
                        os.makedirs(f'{save_path}')

                    save_ckpt(f'{save_path}/g_{self.iter}.pth',
                              [('generator', self.G)],
                              [('optimizer_G', self.optm_G)], self.iter)

        if not os.path.exists(f'{save_path}'):
            os.makedirs(f'{save_path}')
            save_ckpt(f'{save_path}/g_{"final"}.pth', [('generator', self.G)],
                      [('optimizer_G', self.optm_G)], self.iter)

    def test(self, test_loader, result_save_path):
        self.G.eval()  # Fix BN
        for para in self.G.parameters():
            para.requires_grad = False

        count = 0
        for items in test_loader:
            gt_images, masks = self.__cuda__(*items)
            masked_images = gt_images * masks
            masks = torch.cat([masks] * 3, dim=1)
            fake_B = self.G(masked_images, masks)
            comp_B = fake_B * (1 - masks) + gt_images * masks

            if not os.path.exists(f'result/{result_save_path}'):
                os.makedirs(f'result/{result_save_path}')

            if not os.path.exists('result/comp'):
                os.makedirs('result/comp')

            if not os.path.exists('result/real'):
                os.makedirs('result/real')

            if not os.path.exists('result/fake'):
                os.makedirs('result/fake')

            for k in range(comp_B.size(0)):
                count += 1
                grid = make_grid(gt_images[k:k + 1])
                file_path = f'result/real/{count}.png'
                save_image(grid, file_path)

                grid = make_grid(masked_images[k:k + 1])
                file_path = f'result/{result_save_path}/mask_img_{count}.png'
                save_image(grid, file_path)

                grid = make_grid(fake_B[k:k + 1])
                file_path = f'result/{result_save_path}/fake_img_{count}.png'
                save_image(grid, file_path)

                grid = make_grid(comp_B[k:k + 1])
                file_path = f'result/comp/{count}.png'
                save_image(grid, file_path)

    def forward(self, masked_image, mask, gt_image):
        self.real_A = masked_image
        self.real_B = gt_image
        self.mask = mask
        self.fake_B = self.G(masked_image, mask)
        self.comp_B = self.fake_B * (1 - mask) + self.real_B * mask

    def update_parameters(self):
        self.update_G()

    def update_G(self):
        self.optm_G.zero_grad()
        loss_G = self.get_g_loss()
        loss_G.backward()
        self.optm_G.step()

    def get_g_loss(self):
        # fake_A = self.fake_A
        real_B = self.real_B
        fake_B = self.fake_B
        comp_B = self.comp_B

        real_B_feats = self.lossNet(real_B)
        fake_B_feats = self.lossNet(fake_B)
        comp_B_feats = self.lossNet(comp_B)

        tv_loss = self.TV_loss(comp_B * (1 - self.mask))
        style_loss = self.style_loss(real_B_feats,
                                     fake_B_feats) + self.style_loss(
                                         real_B_feats, comp_B_feats)
        preceptual_loss = self.preceptual_loss(
            real_B_feats, fake_B_feats) + self.preceptual_loss(
                real_B_feats, comp_B_feats)
        valid_loss = self.l1_loss(real_B, fake_B, self.mask)
        hole_loss = self.l1_loss(real_B, fake_B, (1 - self.mask))
        # laplace_loss = self.laplace_loss(fake_B, real_B, (1 - self.mask))
        # loss_G = (tv_loss * 0.1 + preceptual_loss * 0.05 + valid_loss * 1 +
        #           hole_loss * 6)
        loss_G = (tv_loss * 0.1 + style_loss * 150 + preceptual_loss * 0.1 +
                  valid_loss * 1 + hole_loss * 6)

        self.l1_loss_val += valid_loss.detach() + hole_loss.detach()
        self.hole_loss += hole_loss.detach()
        # self.laplace_loss_ += laplace_loss.detach()
        self.loss += loss_G.detach()
        # print('loss: ', loss_G.item())
        return loss_G

    def l1_loss(self, f1, f2, mask=1):
        return torch.mean(torch.abs(f1 - f2) * mask)

    def style_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(
            B_feats
        ), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            _, c, w, h = A_feat.size()
            A_feat = A_feat.view(A_feat.size(0), A_feat.size(1),
                                 A_feat.size(2) * A_feat.size(3))
            B_feat = B_feat.view(B_feat.size(0), B_feat.size(1),
                                 B_feat.size(2) * B_feat.size(3))
            A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
            B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
            loss_value += torch.mean(
                torch.abs(A_style - B_style) / (c * w * h))
        return loss_value

    def TV_loss(self, x):
        h_x = x.size(2)
        w_x = x.size(3)
        h_tv = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :]))
        w_tv = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1]))
        return h_tv + w_tv

    def preceptual_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(
            B_feats
        ), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            loss_value += torch.mean(torch.abs(A_feat - B_feat))
        return loss_value

    def laplace(self, A_feats):
        A_feats_left = F.pad(A_feats, [1, 0, 0, 0])[:, :, :, :-1]
        A_feats_right = F.pad(A_feats, [0, 1, 0, 0])[:, :, :, 1:]
        A_feats_top = F.pad(A_feats, [0, 0, 1, 0])[:, :, :-1, :]
        A_feats_bottom = F.pad(A_feats, [0, 0, 0, 1])[:, :, 1:, :]

        result = A_feats_left + A_feats_right + A_feats_top + A_feats_bottom - 4 * A_feats
        return result

    def laplace_loss(self, A_feats, B_feats=0.0, mask=1.0):
        feats_lap = self.laplace(A_feats - B_feats)
        return torch.mean(torch.abs(feats_lap) * mask)

    def __cuda__(self, *args):
        return (item.to(self.device) for item in args)
