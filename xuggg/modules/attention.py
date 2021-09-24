import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextualAttention(nn.Module):
    def __init__(self, patch_size=3, stride=1):
        super(ContextualAttention, self).__init__()
        self.patch_size = patch_size
        self.stride = stride

    def forward(self, foreground):
        bs, ch, h, w = foreground.size()
        foreground_ = foreground.clone()
        conv_kernels_all = foreground_.view(bs, ch, w * h, 1, 1)
        conv_kernels_all = conv_kernels_all.permute(0, 2, 1, 3,
                                                    4)  # (bs, w*h, ch, 1, 1)
        output = []

        for i in range(bs):
            feature_map = foreground[i:i + 1]  # (1, ch, h, w)
            conv_kernels = conv_kernels_all[i] + 1e-7  # (w*h, ch, 1, 1)
            norm_factor = torch.sum(conv_kernels**2, [1, 2, 3],
                                    keepdim=True)**0.5
            conv_kernels = conv_kernels / norm_factor  # (w*h, ch, 1, 1)

            conv_result = F.conv2d(
                feature_map, conv_kernels, padding=self.patch_size //
                2)  # 1x1卷积，每一通道对应的卷积为相应位置的值 (1, w*h, h+2, w+2)
            # result_mean = conv_result.mean(dim=[2, 3], keepdim=True)
            # conv_result = torch.where(
            #     conv_result < result_mean,
            #     torch.tensor([-10.0], device=foreground.device), conv_result)
            attention_scores = F.softmax(conv_result,
                                         dim=1)  # (1, w*h, h+2, w+2)
            # att_score.append(attention_scores)
            feature_map = F.conv_transpose2d(attention_scores,
                                             conv_kernels,
                                             stride=1,
                                             padding=self.patch_size // 2)
            final_output = feature_map
            output.append(final_output)

        return torch.cat(output, dim=0)


class AttentionModule(nn.Module):
    def __init__(self, inchannel, patch_size_list=[1]):
        super(AttentionModule, self).__init__()
        self.att = ContextualAttention(patch_size_list[0])
        self.combiner = nn.Conv2d(inchannel * 2, inchannel, kernel_size=1)

    def forward(self, foreground):
        outputs = self.att(foreground)
        outputs = torch.cat([outputs, foreground], dim=1)
        outputs = self.combiner(outputs)
        return outputs


# class ContextualAttentionMask(nn.Module):
#     def __init__(self, patch_size=3, stride=1):
#         super(ContextualAttentionMask, self).__init__()
#         self.patch_size = patch_size
#         self.stride = stride

#     def forward(self, foreground, mask, background):
#         bs, ch, h, w = foreground.size()
#         foreground_ = foreground.clone()
#         background = background.clone()
#         conv_kernels_all = foreground_.view(bs, ch, w * h, 1, 1)
#         conv_kernels_all = conv_kernels_all.permute(0, 2, 1, 3,
#                                                     4)  # (bs, w*h, ch, 1, 1)
#         conv_kernels_all_ = background.view(bs, ch, w * h, 1, 1)
#         conv_kernels_all_ = conv_kernels_all_.permute(0, 2, 1, 3,
#                                                       4)  # (bs, w*h, ch, 1, 1)
#         output = []

#         for i in range(bs):
#             feature_map = foreground[i:i + 1]  # (1, ch, h, w)
#             feature_map_back = background[i:i + 1]
#             mask_map = mask[i:i + 1, 0:1]  # (1, 1, h, w)
#             mask_map = mask_map.view(h * w, 1, 1, 1)

#             conv_kernels = conv_kernels_all[i] + 1e-7  # (w*h, ch, 1, 1)
#             conv_kernels_ = conv_kernels_all_[i] + 1e-7  # (w*h, ch, 1, 1)

#             norm_factor = torch.sum(conv_kernels**2, [1, 2, 3],
#                                     keepdim=True)**0.5
#             conv_kernels = conv_kernels / norm_factor  # (w*h, ch, 1, 1)

#             norm_factor_ = torch.sum(conv_kernels_**2, [1, 2, 3],
#                                      keepdim=True)**0.5
#             conv_kernels_ = conv_kernels_ / norm_factor_  # (w*h, ch, 1, 1)

#             conv_result = F.conv2d(
#                 feature_map_back, conv_kernels_, padding=self.patch_size //
#                 2)  # 1x1卷积，每一通道对应的卷积为相应位置的值 (1, w*h, h+2, w+2)
#             attention_scores = F.softmax(conv_result,
#                                          dim=1)  # (1, w*h, h+2, w+2)
#             # att_score.append(attention_scores)

#             conv_kernels = conv_kernels * mask_map
#             feature_map_mask = F.conv_transpose2d(attention_scores,
#                                                   conv_kernels,
#                                                   stride=1,
#                                                   padding=self.patch_size // 2)
#             mask_map = mask_map.view(1, 1, h, w)
#             final_output = feature_map_mask * (
#                 1 - mask_map) + feature_map * mask_map
#             output.append(final_output)

#         return torch.cat(output, dim=0)

# class AttentionModuleMask(nn.Module):
#     def __init__(self, inchannel, patch_size_list=[1]):
#         super(AttentionModuleMask, self).__init__()
#         self.att = ContextualAttentionMask(patch_size_list[0])
#         self.combiner = nn.Conv2d(inchannel * 2, inchannel, kernel_size=1)

#     def forward(self, foreground, mask, foreground_):
#         outputs = self.att(foreground, mask, foreground_)
#         outputs = torch.cat([outputs, foreground], dim=1)
#         outputs = self.combiner(outputs)
#         return outputs


class ContextualAttentionMask(nn.Module):
    def __init__(self, patch_size=3, stride=1):
        super(ContextualAttentionMask, self).__init__()
        self.patch_size = patch_size
        self.stride = stride

    def forward(self, foreground, mask):
        bs, ch, h, w = foreground.size()
        background = foreground.clone()
        conv_kernels_all = background.view(bs, ch, w * h, 1, 1)
        conv_kernels_all = conv_kernels_all.permute(0, 2, 1, 3,
                                                    4)  # (bs, w*h, ch, 1, 1)
        output = []

        for i in range(bs):
            feature_map = foreground[i:i + 1]  # (1, ch, h, w)
            mask_map = mask[i:i + 1, 0:1]  # (1, 1, h, w)
            mask_value = torch.sum(mask_map).item()
            if mask_value > (h * w - 10):
                output.append(feature_map)
                continue

            # print("The sum of mask is: ", mask_value, "++++++")
            mask_map = mask_map.view(h * w, 1, 1, 1)
            conv_kernels = conv_kernels_all[i] + 1e-7  # (w*h, ch, 1, 1)
            norm_factor = torch.sum(conv_kernels**2, [1, 2, 3],
                                    keepdim=True)**0.5
            conv_kernels = conv_kernels / norm_factor  # (w*h, ch, 1, 1)

            conv_result = F.conv2d(
                feature_map, conv_kernels, padding=self.patch_size //
                2)  # 1x1卷积，每一通道对应的卷积为相应位置的值 (1, w*h, h+2, w+2)
            attention_scores = F.softmax(conv_result,
                                         dim=1)  # (1, w*h, h+2, w+2)
            # att_score.append(attention_scores)
            conv_kernels = conv_kernels * mask_map
            feature_map_mask = F.conv_transpose2d(attention_scores,
                                                  conv_kernels,
                                                  stride=1,
                                                  padding=self.patch_size // 2)
            mask_map = mask_map.view(1, 1, h, w)
            final_output = feature_map_mask * (
                1 - mask_map) + feature_map * mask_map
            output.append(final_output)

        return torch.cat(output, dim=0)


class AttentionModuleMask(nn.Module):
    def __init__(self, inchannel, patch_size_list=[1]):
        super(AttentionModuleMask, self).__init__()
        self.att = ContextualAttentionMask(patch_size_list[0])
        self.combiner = nn.Conv2d(inchannel * 2, inchannel, kernel_size=1)

    def forward(self, foreground, mask, background):
        outputs = self.att(background, mask)
        outputs = torch.cat([outputs, foreground], dim=1)
        outputs = self.combiner(outputs)
        return outputs


class AttentionExtractorMask(nn.Module):
    def __init__(self, patch_size=3, stride=1):
        super(AttentionExtractorMask, self).__init__()
        self.patch_size = patch_size
        self.stride = stride

    def forward(self, foreground):
        bs, ch, h, w = foreground.size()
        foreground_ = foreground.clone()
        conv_kernels_all = foreground_.view(bs, ch, w * h, 1, 1)
        conv_kernels_all = conv_kernels_all.permute(0, 2, 1, 3,
                                                    4)  # (bs, w*h, ch, 1, 1)
        output = []

        for i in range(bs):
            feature_map = foreground[i:i + 1]  # (1, ch, h, w)
            conv_kernels = conv_kernels_all[i] + 1e-7  # (w*h, ch, 1, 1)
            norm_factor = torch.sum(conv_kernels**2, [1, 2, 3],
                                    keepdim=True)**0.5
            conv_kernels = conv_kernels / norm_factor  # (w*h, ch, 1, 1)
            conv_result = F.conv2d(
                feature_map, conv_kernels,
                padding=(self.patch_size //
                         2))  # 1x1卷积，每一通道对应的卷积为相应位置的值 (1, w*h, h+2, w+2)
            attention_scores = F.softmax(conv_result,
                                         dim=1)  # (1, w*h, h+2, w+2)
            output.append(attention_scores)

        return torch.cat(output, dim=0)


class AttentionMergeMask(nn.Module):
    def __init__(self, inchannel, patch_size=3, stride=1):
        super(AttentionMergeMask, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.combiner = nn.Conv2d(inchannel * 2, inchannel, kernel_size=1)

    def attention_transfer(self, foreground, mask, attention_scores):
        bs, ch, h, w = foreground.size()
        foreground_ = foreground.clone()
        conv_kernels_all = foreground_.view(bs, ch, w * h, 1, 1)
        conv_kernels_all = conv_kernels_all.permute(0, 2, 1, 3,
                                                    4)  # (bs, w*h, ch, 1, 1)
        output = []

        for i in range(bs):
            feature_map = foreground[i:i + 1]  # (1, ch, h, w)
            mask_map = mask[i:i + 1, 0:1]  # (1, 1, h, w)
            # mask_value = torch.sum(mask_map).item()
            # if mask_value > (h * w - 10):
            #     output.append(feature_map)
            #     continue

            # print("The sum of mask is: ", mask_value, "++++++")
            mask_map = mask_map.view(h * w, 1, 1, 1)
            conv_kernels = conv_kernels_all[i] + 1e-7  # (w*h, ch, 1, 1)
            norm_factor = torch.sum(conv_kernels**2, [1, 2, 3],
                                    keepdim=True)**0.5
            conv_kernels = conv_kernels / norm_factor  # (w*h, ch, 1, 1)

            conv_kernels = conv_kernels * mask_map
            feature_map_mask = F.conv_transpose2d(attention_scores[i:i + 1],
                                                  conv_kernels,
                                                  stride=1,
                                                  padding=self.patch_size // 2)
            mask_map = mask_map.view(1, 1, h, w)
            final_output = feature_map_mask * (
                1 - mask_map) + feature_map * mask_map
            output.append(final_output)

        return torch.cat(output, dim=0)

    def forward(self, foreground, mask, attention_scores):
        att_output = self.attention_transfer(foreground, mask,
                                             attention_scores)
        outputs = torch.cat([foreground, att_output], dim=1)
        outputs = self.combiner(outputs)
        return outputs
