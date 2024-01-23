import torch
from tqdm import tqdm

from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion

from utils import AverageMeter
from logger import logger



class DETRModel(torch.nn.Module):
    def __init__(self, num_classes, num_queries):
        """
        Constructor
        Params:
            num_classes (int): number of classes that the model is able to predict
            num_queries (int): number of bounding boxes per image
        """
        super(DETRModel,self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)

        self.in_features = self.model.class_embed.in_features
        self.model.class_embed = torch.nn.Linear(in_features=self.in_features,
                                                 out_features=self.num_classes)
        self.model.num_queries = self.num_queries
        
    def forward(self,images):
        return self.model(images)


class TrainingPipeline():
    def __init__(self, model, optimizer, loss, training_dataloader, test_dataloader) -> None:
        """
        Constructor, also initializes the HungarianMatcher, a weights dictionary and a list with 
        the losses
        Params:
            model:
            optimizer:
            criterion:
            training_dataloader:
            test_dataloader:
        """
        # Device
        self.device =  'cuda' if torch.cuda.is_available() else 'cpu'

        # Variables related to the forward pass
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss = loss.to(self.device)

        # Datasets
        self.training_dataloader = training_dataloader
        self.test_dataloader = test_dataloader

        # utilities
        self.matcher = HungarianMatcher()
        self.weight_dict = {'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1}
        self.losses = ["losses", "boxes", "carinality"]

    def train(self, epoch: int, scheduler=None, batch_size=32):
        """
        Performs a forward pass over the class model
        Params:
            epoch (int)
        """
        self.model.train()
        self.loss.train()

        summary_loss = AverageMeter()

        # initializes a progress bar for the training loop.
        tk0 = tqdm(self.training_dataloader, total=len(self.training_dataloader))

        # iterates over the training_dataloader
        for step, (images, targets, image_ids) in enumerate(tk0):

            # Moves the images and the targets to the class device
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # computes the predictions
            predictions = self.model(images)
            
            # computes the loss
            loss_dict = self.loss(predictions, targets)

            # computes the loss
            weight_dict = self.loss.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # clears and computes the gradient of the loss with respect to the model parameters.
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            if scheduler:
                scheduler.step()
            
            summary_loss.update(losses.item(), batch_size)
            tk0.set_postfix(loss=summary_loss.avg)

            # Logging
            if step % 100 == 0:  # Log every 100 steps
                logger.info(f'Epoch [{epoch+1}], Step [{step+1}/{len(self.training_dataloader)}], 
                            Image ID: {image_ids[0]}, Loss: {losses.item():.4f}')
            
        return summary_loss



