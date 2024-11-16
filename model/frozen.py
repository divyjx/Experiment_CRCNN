import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
import torchvision.transforms as T
from pycocotools.cocoeval import COCOeval

# **Step 1: Define Model with Frozen Backbone and RPN, Unfreeze FC layers of ROI head**
def get_faster_rcnn_model(num_classes):
    # Load pretrained Faster R-CNN with ResNet-50 backbone
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Freeze the backbone and RPN layers
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.rpn.parameters():
        param.requires_grad = False
    
    # Freeze all layers in the ROI heads except the FC layers
    for name, param in model.roi_heads.named_parameters():
        if "box_head.fc" not in name:
            param.requires_grad = False

    # Replace the classifier with a new head with the correct number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

# **Step 2: Define Optimizer and Scheduler**
def get_optimizer_scheduler(model):
    # Only the FC layers of ROI head are trainable, so we use filter() to optimize only those
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=0.001, 
        momentum=0.9, 
        weight_decay=0.0004
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    return optimizer, scheduler

# **Step 3: Training Function**
def train_model(model, data_loader, optimizer, scheduler, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
        
        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

# **Step 4: Evaluation Function (COCO-style mAP)**
def evaluate_model(model, data_loader):
    model.eval()
    coco_eval = COCOeval(...)  # Initialize with your dataset's COCO annotations
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        with torch.no_grad():
            outputs = model(images)
            # Convert outputs to COCO format and evaluate
            predictions = convert_to_coco_format(outputs, targets)
            coco_eval.add(predictions)
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    mAP = coco_eval.stats[0]  # mAP@[0.5:0.95]
    return mAP

# **Step 5: Main Training Stages**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 16  # 15 classes + 'empty' class
model = get_faster_rcnn_model(num_classes).to(device)
optimizer, scheduler = get_optimizer_scheduler(model)

# Data loaders for train, cis val, trans val, cis test, trans test
train_loader = ...          # Define DataLoader for training set
cis_val_loader = ...        # Define DataLoader for cis validation set
trans_val_loader = ...      # Define DataLoader for trans validation set
cis_test_loader = ...       # Define DataLoader for cis test set
trans_test_loader = ...     # Define DataLoader for trans test set

# **Stage 1: Fine-tune on training set**
print("Stage 1: Fine-tuning on Training Set")
train_model(model, train_loader, optimizer, scheduler, num_epochs=10)

# **Stage 2: Fine-tune on training + validation set**
print("Stage 2: Fine-tuning on Training + Validation Set")
combined_train_val_loader = DataLoader(
    train_loader.dataset + cis_val_loader.dataset + trans_val_loader.dataset,
    batch_size=4, shuffle=True
)
train_model(model, combined_train_val_loader, optimizer, scheduler, num_epochs=5)

# **Stage 3: Evaluation**
print("Evaluating on Cis and Trans Test Sets")
mAP_cis = evaluate_model(model, cis_test_loader)
mAP_trans = evaluate_model(model, trans_test_loader)

print(f"mAP on Cis Test Set: {mAP_cis:.4f}")
print(f"mAP on Trans Test Set: {mAP_trans:.4f}")
