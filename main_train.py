import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import Trainer

from ORSI_SOD_dataset import ORSI_SOD_dataset
from src.HFANet_R import net as Net_R
# from src.HFANet_V import net as Net_V

from src.TSFANet import net as Net_V

from src.Loss import total_loss

from metric import metrics
import os
import csv

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SaliencyModel(pl.LightningModule):
    def __init__(self, model_type="HFANet_R", dataset_name="ORSSD", save_dir='./saved_models'):
        super(SaliencyModel, self).__init__()
        self.save_dir = save_dir  # 模型保存路径
        os.makedirs(self.save_dir, exist_ok=True)

        if model_type == "HFANet_R":
            self.model = Net_R().to(device)
            # if dataset_name == "ORSSD":
            #     self.model.load_state_dict(torch.load("./HFANet_R_weights/ORSSD_weights.pth", map_location='cuda:0'))
            #     # self.model.load_state_dict(torch.load("./HFANet_R_weights/ORSSD_weights.pth", map_location='cuda:1'))
            # elif dataset_name == "EORSSD":
            #     self.model.load_state_dict(torch.load("./HFANet_R_weights/EORSSD_weights.pth", map_location='cuda:0'))
            #     # self.model.load_state_dict(torch.load("./HFANet_R_weights/EORSSD_weights.pth", map_location='cuda:1'))
            # elif dataset_name == "ors-4199":
            #     self.model.load_state_dict(torch.load("./HFANet_R_weights/ORS_4199_weights.pth", map_location='cuda:0'))
            #     # self.model.load_state_dict(torch.load("./HFANet_R_weights/ORS_4199_weights.pth", map_location='cuda:1'))
        elif model_type == "HFANet_V":
            self.model = Net_V().to(device)
            # if dataset_name == "ORSSD":
            #     self.model.load_state_dict(torch.load("./HFANet_V_weights/ORSSD_weights.pth", map_location='cuda:0'))
            #     # self.model.load_state_dict(torch.load("./HFANet_V_weights/ORSSD_weights.pth", map_location='cuda:1'))
            # elif dataset_name == "EORSSD":
            #     self.model.load_state_dict(torch.load("./HFANet_V_weights/EORSSD_weights.pth", map_location='cuda:0'))
            #     # self.model.load_state_dict(torch.load("./HFANet_V_weights/EORSSD_weights.pth", map_location='cuda:1'))
            # elif dataset_name == "ors-4199":
            #     self.model.load_state_dict(torch.load("./HFANet_V_weights/ORS_4199_weights.pth", map_location='cuda:0'))
            #     # self.model.load_state_dict(torch.load("./HFANet_V_weights/ORS_4199_weights.pth", map_location='cuda:1'))
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.criterion = total_loss
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.metric_evaluator = metrics()
        self.header_written = False

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, label, edge, image_id = batch
        inputs, label, edge = inputs.to(device), label.to(device), edge.to(device)
        smap1, smap2, smap3, smap4, smap5, edge1, edge2, edge3, edge4, edge5 = self(inputs)

        loss = self.bce_loss(smap1, label)
        # loss = self.criterion([smap1, smap2, smap3, smap4, smap5], [edge1, edge2, edge3, edge4, edge5], label, edge, lambdas=(1, 1))
        self.log('train_loss', loss)
        # print(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, label, edge, image_id = batch
        inputs, label, edge = inputs.to(device), label.to(device), edge.to(device)
        smap1, smap2, smap3, smap4, smap5, edge1, edge2, edge3, edge4, edge5 = self(inputs)

        loss = self.bce_loss(smap1, label)
        # loss = self.criterion([smap1, smap2, smap3, smap4, smap5], [edge1, edge2, edge3, edge4, edge5], label, edge,
        #                       lambdas=(1, 1))
        self.log('val_loss', loss)

        pred = smap1.sigmoid()
        pred = pred.cpu().numpy().squeeze()
        label = label.cpu().numpy().squeeze()

        self.metric_evaluator.add_batch(pred=pred, mask=label, threshold=0.5)
        # print(loss)
        return loss

    def validation_epoch_end(self, outputs):
        metric_results = self.metric_evaluator.show()
        self.log('val_metrics', metric_results)
        print(f"Validation Metrics: {metric_results}")

        csv_file_path = os.path.join(self.save_dir, 'metrics.csv')
        record = {
            'epoch': self.current_epoch,
            'loss': sum(outputs) / len(outputs),
            'metrics': metric_results
        }
        self.save_metrics_to_csv(csv_file_path, record)

        self.metric_evaluator.reset()

        self.save_model()

    def save_metrics_to_csv(self, file_path, record):
        fieldnames = ['Epoch', 'Loss', 'Acc', 'Precision', 'Recall', 'MIoU', 'mDice', 'F1', 'MAE']

        if not self.header_written:
            with open(file_path, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                self.header_written = True

        with open(file_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            row = {
                'Epoch': record['epoch'],
                'Loss': record['loss'],
                'Acc': record['metrics'].get('accuracy', '0.0'),
                'Precision': record['metrics'].get('precision', '0.0'),
                'Recall': record['metrics'].get('recall', '0.0'),
                'MIoU': record['metrics'].get('mIoU', '0.0'),
                'mDice': record['metrics'].get('mDice', '0.0'),
                'F1': record['metrics'].get('f1-score', '0.0'),
                'MAE': record['metrics'].get('MAE', '0.0'),
            }
            writer.writerow(row)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        return [optimizer], [scheduler]

    def save_model(self):
        """保存模型权重到指定目录"""
        model_save_path = os.path.join(self.save_dir, f'epoch_{self.current_epoch}.pth')
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")


def get_dataloader(dataset_name, batch_size=8, num_workers=4, mode="train"):
    dataset_root = f"../PYcharm/image_segment/dataset/{dataset_name}/{mode}/"
    dataset = ORSI_SOD_dataset(root=dataset_root, mode=mode, aug=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=(mode == "train"))
    return dataloader

if __name__ == "__main__":
    model_type = "HFANet_V"
    dataset_name = "data"

    train_loader = get_dataloader(dataset_name, batch_size=8, mode="train")
    val_loader = get_dataloader(dataset_name, batch_size=8, mode="test")

    model = SaliencyModel(model_type=model_type, dataset_name=dataset_name, save_dir='./saved_models')
    trainer = Trainer(max_epochs=500, accelerator='gpu', devices=[0], gpus=[0])

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
