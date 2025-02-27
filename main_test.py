import argparse
import torch
import os
from torch.utils.data import DataLoader
from ORSI_SOD_dataset import ORSI_SOD_dataset
from src.HFANet_R import net as Net_R
# from src.HFANet_V import net as Net_V

# from src.HFANet_V_Mamba import net as Net_V
# from src.HFANet_V_Mamba_corr import net as Net_V
from src.HFANet_V_Mamba_corr_fuse import net as Net_V


from metric import metrics
import cv2
import numpy as np
from tqdm import tqdm  # 导入 tqdm 库

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description='Saliency Model Testing')
    parser.add_argument('--model_type', type=str, default='HFANet_V', choices=['HFANet_R', 'HFANet_V'])
    parser.add_argument('--dataset_name', type=str, default='ors-4199')
    parser.add_argument('--weight_epoch', type=int, default=213)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--threshold', type=float, default=0.5)
    return parser.parse_args()

def test_model(model, test_loader, threshold=0.5, save_dir='./test_results'):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    metric_evaluator = metrics()

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Processing Batches")):  # 使用 tqdm 包装 DataLoader
            inputs, label, edge, image_id = batch
            inputs, label, edge = inputs.to(device), label.to(device), edge.to(device)
            smap1, smap2, smap3, smap4, smap5, edge1, edge2, edge3, edge4, edge5 = model(inputs)
            pred = smap1.sigmoid().cpu().numpy()

            for j in range(pred.shape[0]):
                pred_np = pred[j].squeeze()
                binary_pred = (pred_np > threshold).astype(np.uint8)
                save_path = os.path.join(save_dir, f"{image_id[j]}.png")
                cv2.imwrite(save_path, binary_pred * 255)
                label_np = label[j].cpu().numpy().squeeze()
                metric_evaluator.add_batch(pred=pred_np, mask=label_np, threshold=threshold)

    metric_results = metric_evaluator.show()
    print(f"Test Metrics: {metric_results}")
    metric_evaluator.reset()

def get_dataloader(dataset_name, batch_size=8, num_workers=4, mode="test"):
    # dataset_root = f"../../dataset/{dataset_name}/{mode}/"
    dataset_root = f"../PYcharm/image_segment/dataset/{dataset_name}/{mode}/"

    dataset = ORSI_SOD_dataset(root=dataset_root, mode=mode, aug=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return dataloader

if __name__ == "__main__":
    args = parse_args()
    if args.model_type == "HFANet_R":
        model = Net_R().to(device)
    elif args.model_type == "HFANet_V":
        model = Net_V().to(device)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    weights_path = f'./saved_models/epoch_{args.weight_epoch}.pth'
    model.load_state_dict(torch.load(weights_path, map_location=device))
    test_loader = get_dataloader(args.dataset_name, batch_size=args.batch_size, num_workers=args.num_workers, mode="test")
    test_model(model, test_loader, threshold=args.threshold, save_dir=args.save_dir)
