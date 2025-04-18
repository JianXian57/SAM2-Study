#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Copyright (c) Meta Platforms, Inc. and affiliates.


# # Object masks in images from prompts with SAM 2
# # SAM 2 提示中图像中的对象掩码

# Segment Anything Model 2 (SAM 2) predicts object masks given prompts that indicate the desired object. The model first converts the image into an image embedding that allows high quality masks to be efficiently produced from a prompt. 
# 
# The `SAM2ImagePredictor` class provides an easy interface to the model for prompting the model. It allows the user to first set an image using the `set_image` method, which calculates the necessary image embeddings. Then, prompts can be provided via the `predict` method to efficiently predict masks from those prompts. The model can take as input both point and box prompts, as well as masks from the previous iteration of prediction.
# 
# Segment Anything Model 2 （SAM 2） 在给定指示所需对象的提示中预测对象掩码。该模型首先将图像转换为图像嵌入，以便从提示中高效生成高质量的掩码。
# 
# 'SAM2ImagePredictor' 类为模型提供了一个简单的接口，用于提示模型。它允许用户首先使用 'set_image' 方法设置图像，该方法计算必要的图像嵌入。然后，可以通过 'predict' 方法提供提示，以有效地从这些提示中预测掩码。该模型可以将点提示和框提示以及上一次预测迭代的掩码作为输入。

# <a target="_blank" href="https://colab.research.google.com/github/facebookresearch/sam2/blob/main/notebooks/image_predictor_example.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# ## Environment Set-up
# ## 环境设置

# If running locally using jupyter, first install `sam2` in your environment using the [installation instructions](https://github.com/facebookresearch/sam2#installation) in the repository.
# 
# If running from Google Colab, set `using_colab=True` below and run the cell. In Colab, be sure to select 'GPU' under 'Edit'->'Notebook Settings'->'Hardware accelerator'. Note that it's recommended to use **A100 or L4 GPUs when running in Colab** (T4 GPUs might also work, but could be slow and might run out of memory in some cases).
# 
# 如果使用 jupyter 在本地运行，请首先使用存储库中的 [安装说明]（https://github.com/facebookresearch/sam2#installation） 在您的环境中安装“sam2”。
# 
# 如果从 Google Colab 运行，请在下方设置 'using_colab=True' 并运行单元格。在 Colab 中，请务必在“编辑”->“笔记本设置”->“硬件加速器”下选择“GPU”。请注意，在 Colab 中运行时，建议使用 **A100 或 L4 GPU**（T4 GPU 也可能有效，但可能会很慢，并且在某些情况下可能会耗尽内存）。

# In[2]:


using_colab = False


# In[3]:


if using_colab:
    import torch
    import torchvision
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    import sys
    get_ipython().system('{sys.executable} -m pip install opencv-python matplotlib')
    get_ipython().system("{sys.executable} -m pip install 'git+https://github.com/facebookresearch/sam2.git'")

    get_ipython().system('mkdir -p images')
    get_ipython().system('wget -P images https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/truck.jpg')
    get_ipython().system('wget -P images https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/groceries.jpg')

    get_ipython().system('mkdir -p ../checkpoints/')
    get_ipython().system('wget -P ../checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt')


# ## Set-up

# Necessary imports and helper functions for displaying points, boxes, and masks.
# 用于显示点、框和掩码的必要导入和辅助函数。

# In[4]:


import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image


# In[5]:


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


# In[6]:


np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


# ## Example image

# In[7]:


image = Image.open('images/truck.jpg')
image = np.array(image.convert("RGB"))


# In[8]:


plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('on')
plt.show()


# ## Selecting objects with SAM 2

# First, load the SAM 2 model and predictor. Change the path below to point to the SAM 2 checkpoint. Running on CUDA and using the default model are recommended for best results.
# 
# 首先，加载 SAM 2 模型和预测变量。将下面的路径更改为指向 SAM 2 检查点。建议在 CUDA 上运行并使用默认模型以获得最佳结果。

# In[9]:


from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)


# Process the image to produce an image embedding by calling `SAM2ImagePredictor.set_image`. `SAM2ImagePredictor` remembers this embedding and will use it for subsequent mask prediction.
# 
# 通过调用 'SAM2ImagePredictor.set_image' 处理图像以生成图像嵌入。'SAM2ImagePredictor' 会记住此嵌入，并将其用于后续的掩码预测。

# In[10]:


predictor.set_image(image)


# To select the truck, choose a point on it. Points are input to the model in (x,y) format and come with labels 1 (foreground point) or 0 (background point). Multiple points can be input; here we use only one. The chosen point will be shown as a star on the image.
# 
# 要选择卡车，请在卡车上选择一个点。点以 （x，y） 格式输入到模型中，并带有标签 1（前景点）或 0（背景点）。可以输入多个点;这里我们只使用一个。所选点将在图像上显示为星号。

# In[11]:


input_point = np.array([[500, 375]])
input_label = np.array([1])


# In[12]:


plt.figure(figsize=(10, 10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()  


# In[13]:


print(predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)


# Predict with `SAM2ImagePredictor.predict`. The model returns masks, quality predictions for those masks, and low resolution mask logits that can be passed to the next iteration of prediction.
# 
# 使用 'SAM2ImagePredictor.predict' 进行预测。该模型将返回掩码、这些掩码的质量预测以及可传递给下一次预测迭代的低分辨率掩码 logit。

# In[14]:


masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
sorted_ind = np.argsort(scores)[::-1]
masks = masks[sorted_ind]
scores = scores[sorted_ind]
logits = logits[sorted_ind]


# With `multimask_output=True` (the default setting), SAM 2 outputs 3 masks, where `scores` gives the model's own estimation of the quality of these masks. This setting is intended for ambiguous input prompts, and helps the model disambiguate different objects consistent with the prompt. When `False`, it will return a single mask. For ambiguous prompts such as a single point, it is recommended to use `multimask_output=True` even if only a single mask is desired; the best single mask can be chosen by picking the one with the highest score returned in `scores`. This will often result in a better mask.
# 
# 使用 'multimask_output=True'（默认设置），SAM 2 输出 3 个掩码，其中 'scores' 给出模型自己对这些掩码质量的估计。此设置适用于不明确的输入提示，并帮助模型消除与提示一致的不同对象的歧义。当 'False' 时，它将返回一个掩码。对于不明确的提示（例如单个点），即使只需要一个掩码，也建议使用 'multimask_output=True';可以通过选择在 'scores' 中返回最高分数的掩码来选择最佳单个掩码。这通常会产生更好的蒙版。

# In[15]:


masks.shape  # (number_of_masks) x H x W


# In[16]:


show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)


# ## Specifying a specific object with additional points
# ## 使用附加点指定特定对象

# The single input point is ambiguous, and the model has returned multiple objects consistent with it. To obtain a single object, multiple points can be provided. If available, a mask from a previous iteration can also be supplied to the model to aid in prediction. When specifying a single object with multiple prompts, a single mask can be requested by setting `multimask_output=False`.
# 
# 单个输入点不明确，并且模型已返回多个与其一致的对象。要获取单个对象，可以提供多个点。如果可用，还可以向模型提供先前迭代的掩码以帮助预测。当指定具有多个提示的单个对象时，可以通过设置 'multimask_output=False' 来请求单个掩码。

# In[17]:


input_point = np.array([[500, 375], [1125, 625]])
input_label = np.array([1, 1])

mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask


# In[18]:


masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    mask_input=mask_input[None, :, :],
    multimask_output=False,
)


# In[19]:


masks.shape


# In[20]:


show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label)


# To exclude the car and specify just the window, a background point (with label 0, here shown in red) can be supplied.
# 
# 要排除汽车并仅指定窗口，可以提供背景点（标签为 0，此处以红色显示）。

# In[21]:


input_point = np.array([[500, 375], [1125, 625]])
input_label = np.array([1, 0])

mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask


# In[22]:


masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    mask_input=mask_input[None, :, :],
    multimask_output=False,
)


# In[23]:


show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label)


# ## Specifying a specific object with a box
# ## 用 box 指定一个特定的对象

# The model can also take a box as input, provided in xyxy format.
# 该模型还可以将框作为输入，以 xyxy 格式提供。

# In[24]:


input_box = np.array([425, 600, 700, 875])


# In[25]:


masks, scores, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,
)


# In[26]:


show_masks(image, masks, scores, box_coords=input_box)


# ## Combining points and boxes
# ## 组合点和方框

# Points and boxes may be combined, just by including both types of prompts to the predictor. Here this can be used to select just the trucks's tire, instead of the entire wheel.
# 点和框可以组合在一起，只需将两种类型的提示都包含到预测器中即可。在这里，这可用于仅选择卡车的轮胎，而不是整个车轮。

# In[27]:


input_box = np.array([425, 600, 700, 875])
input_point = np.array([[575, 750]])
input_label = np.array([0])


# In[28]:


masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    box=input_box,
    multimask_output=False,
)


# In[29]:


show_masks(image, masks, scores, box_coords=input_box, point_coords=input_point, input_labels=input_label)


# ## Batched prompt inputs
# ## 批量提示输入

# `SAM2ImagePredictor` can take multiple input prompts for the same image, using `predict` method. For example, imagine we have several box outputs from an object detector.
# 'SAM2ImagePredictor' 可以使用 'predict' 方法为同一张图像获取多个输入提示。例如，假设我们有多个来自对象检测器的盒子输出。

# In[30]:


input_boxes = np.array([
    [75, 275, 1725, 850],
    [425, 600, 700, 875],
    [1375, 550, 1650, 800],
    [1240, 675, 1400, 750],
])


# In[31]:


masks, scores, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)


# In[32]:


masks.shape  # (batch_size) x (num_predicted_masks_per_input) x H x W


# In[33]:


plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask in masks:
    show_mask(mask.squeeze(0), plt.gca(), random_color=True)
for box in input_boxes:
    show_box(box, plt.gca())
plt.axis('off')
plt.show()


# ## End-to-end batched inference
# If all prompts are available in advance, it is possible to run SAM 2 directly in an end-to-end fashion. This also allows batching over images.
# 
# ## 端到端批量推理
# 如果所有提示都提前可用，则可以直接以端到端方式运行 SAM 2。这也允许对图像进行批处理。

# In[34]:


image1 = image  # truck.jpg from above
image1_boxes = np.array([
    [75, 275, 1725, 850],
    [425, 600, 700, 875],
    [1375, 550, 1650, 800],
    [1240, 675, 1400, 750],
])

image2 = Image.open('images/groceries.jpg')
image2 = np.array(image2.convert("RGB"))
image2_boxes = np.array([
    [450, 170, 520, 350],
    [350, 190, 450, 350],
    [500, 170, 580, 350],
    [580, 170, 640, 350],
])

img_batch = [image1, image2]
boxes_batch = [image1_boxes, image2_boxes]


# In[35]:


predictor.set_image_batch(img_batch)


# In[36]:


masks_batch, scores_batch, _ = predictor.predict_batch(
    None,
    None, 
    box_batch=boxes_batch, 
    multimask_output=False
)


# In[37]:


for image, boxes, masks in zip(img_batch, boxes_batch, masks_batch):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)   
    for mask in masks:
        show_mask(mask.squeeze(0), plt.gca(), random_color=True)
    for box in boxes:
        show_box(box, plt.gca())


# Similarly, we can have a batch of point prompts defined over a batch of images
# 同样，我们可以在一批图像上定义一批点提示

# In[38]:


image1 = image  # truck.jpg from above
image1_pts = np.array([
    [[500, 375]],
    [[650, 750]]
    ]) # Bx1x2 where B corresponds to number of objects 
image1_labels = np.array([[1], [1]])

image2_pts = np.array([
    [[400, 300]],
    [[630, 300]],
])
image2_labels = np.array([[1], [1]])

pts_batch = [image1_pts, image2_pts]
labels_batch = [image1_labels, image2_labels]


# In[39]:


masks_batch, scores_batch, _ = predictor.predict_batch(pts_batch, labels_batch, box_batch=None, multimask_output=True)

# Select the best single mask per object
best_masks = []
for masks, scores in zip(masks_batch,scores_batch):
    best_masks.append(masks[range(len(masks)), np.argmax(scores, axis=-1)])


# In[40]:


for image, points, labels, masks in zip(img_batch, pts_batch, labels_batch, best_masks):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)   
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    show_points(points, labels, plt.gca())

