# Install detr: pip install detr
from detr.models.matcher import HungarianMatcher
from detr.models.detr    import SetCriterion

"""
Usage:
# forward pass
logits, pred_boxes = model(images, …)

# build targets list of dicts:
#   targets[i] = {
#     'labels': tensor of shape (M_i,),
#     'boxes':  tensor of shape (M_i,4)  # in normalized [0,1] cx,cy,w,h
#   }

loss_dict = criterion(
    {
      'pred_logits': logits,       # (batch_size, N, C+1)
      'pred_boxes':  pred_boxes    # (batch_size, N,   4)
    },
    targets
)
loss = sum(loss_dict[k] * 1.0 for k in loss_dict)  # already weighted
loss.backward()
optimizer.step()
"""


matcher = HungarianMatcher(
    cost_class=1.0,
    cost_bbox=5.0,
    cost_giou=2.0
)

weight_dict = {
    'loss_ce':   1.0,   # classification
    'loss_bbox': 5.0,   # L1
    'loss_giou': 2.0,   # GIoU
}
eos_coef = 0.1  # weight for no-object class

def get_loss_function(num_classes=365):
    criterion = SetCriterion(
        num_classes=num_classes+1,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=eos_coef,
        losses=['labels','boxes']
    )
    return criterion