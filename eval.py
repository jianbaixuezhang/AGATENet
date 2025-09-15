import os
import torch
from torchvision.transforms import functional as F
from data import test_dataloader


def _eval(model, args):
    state_dict = torch.load(args.test_model, weights_only=True)
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=4)
    torch.cuda.empty_cache()
    model.eval()
    factor = 8

    with torch.no_grad():
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data
            input_img = input_img.to(device)
            h, w = input_img.shape[2], input_img.shape[3]
            H = ((h + factor - 1) // factor) * factor
            W = ((w + factor - 1) // factor) * factor
            input_img_padded = F.pad(input_img, padding=(0, W - w, 0, H - h), padding_mode='reflect')
            pred = model(input_img_padded)[:, :, :h, :w]
            pred_clip = torch.clamp(pred, 0, 1)

            if args.save_image:
                save_name = os.path.join(args.result_dir, name[0])
                pred_pil = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred_pil.save(save_name)

            print(f'{iter_idx + 1} iter saved: {name[0]}')

