from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
from scripts.model import QVGGNetCAM
import cv2 as cv


# parameter
num_classes = 2
num_epoch = 20
batch_size = 32
learning_rate = 0.001
image_size = 224

train_data_path = '../dataset/testset/train'
test_data_path = '../dataset/testset/test'
save_path = '../backup/new_model.pth'

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])


# data loading
train_datasets = datasets.ImageFolder(root=train_data_path, transform=transform)
train_dataloader = DataLoader(dataset=train_datasets, batch_size=batch_size, shuffle=True)

test_datasets = datasets.ImageFolder(root=test_data_path, transform=transform)
test_dataloader = DataLoader(dataset=test_datasets, batch_size=batch_size, shuffle=False)


# load model
model = QVGGNetCAM(num_classes=num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_func = torch.nn.CrossEntropyLoss()

# # training
# loss_mean_arr = []
# val_loss_mean_arr = []
#
# for epoch in range(num_epoch):
#     model.train()
#     loss_arr = []
#
#     for batch, data in enumerate(train_dataloader):
#         x, label = data
#         if torch.cuda.is_available():
#             x = x.cuda()
#             label = label.cuda()
#
#         output, feature = model(x)
#
#         optim.zero_grad()
#         loss = loss_func(output, label)
#
#         loss.backward()
#         optim.step()
#
#         print('epoch [{}/{}] | batch [{}/{}] | loss : {:.4f}'.format(
#             epoch, num_epoch, batch, len(train_dataloader), loss.item()))
#
#         loss_arr.append(loss.item())
#
#     print('epoch [{}/{}] | loss mean : {:.4f}'.format(
#         epoch, num_epoch, np.mean(loss_arr)))
#
#     loss_mean_arr.append(np.mean(loss_arr))
#
#     model.eval()
#     with torch.no_grad():
#         val_loss_arr = []
#         for batch, data in enumerate(test_dataloader):
#             x, label = data
#             if torch.cuda.is_available():
#                 x = x.cuda()
#                 label = label.cuda()
#
#             output, feature = model(x)
#             val_loss = loss_func(output, label)
#             val_loss_arr.append(val_loss.item())
#
#         print('epoch [{}/{}] | val loss mean : {:.4f}'.format(
#             epoch, num_epoch, np.mean(val_loss_arr)))
#
#         val_loss_mean_arr.append(np.mean(val_loss_arr))
#
#
# # model save
# torch.save(model, '../backup/new_model.pth')

# load save model
load_model = torch.load('../backup/new_model.pth')
load_model.eval()

# 모델의 state_dict 출력
print(load_model.state_dict())
for i in load_model.state_dict():
    print(i)

key = '16.1.weight'
param = load_model.module.feature.state_dict()[key]
print('param.shape: ', param.shape)
print('param.requires_grad: ', param.requires_grad)

test_dataloader = DataLoader(dataset=test_datasets, batch_size=1, shuffle=False)

# cal accuracy & CAM
total_num = 0
correct_num = 0

for data, label in test_dataloader:
    if torch.cuda.is_available():
        data = data.cuda()
        label = label.cuda()

    output, features = load_model(data)
    weights = load_model.module.feature.state_dict()[key]

    s = torch.zeros(size=features[0, 0].shape)
    for index in range(len(weights)):
        f = features[0, index]
        w = weights[index]
        m = f * w
        s += m.to('cpu')

    image_np = data.to('cpu')
    image_np = torch.squeeze(image_np[0]).numpy()
    image_np = np.transpose(image_np, (1, 2, 0))
    image_np = cv.normalize(image_np, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
    image_np = cv.cvtColor(image_np, cv.COLOR_BGR2GRAY)
    image_pil = transforms.ToPILImage()(image_np)

    s_np = s.detach().numpy()
    s_np = cv.resize(s_np, dsize=(224, 224), interpolation=cv.INTER_CUBIC)
    s_np = cv.normalize(s_np, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
    s_image = transforms.ToPILImage()(s_np)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image_pil, cmap='gray')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(image_pil, cmap='gray', alpha=1.0)
    ax2.imshow(s_image, cmap='jet', alpha=0.5)

    plt.show()

    _, predict = torch.max(output.data, 1)

    total_num += label.size(0)
    correct_num += (predict == label).sum().item()

print("Accuracy of Test Data : {:.2f} %".format(100 * correct_num / total_num))

