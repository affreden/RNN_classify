"""
@Time    : 2020/12/1 16:32
@Author  : Affreden
@Email   : affreden@gmail.com
@File    : classification.py
"""
import torch
import random

classes = {}
train_x = []
train_y = []
data_dir = "soybean-large.data.txt"
stored_data = open(data_dir, 'r').readlines()
pre_train_x = []
pre_train_y = []


def encode(name):
    return all_classes.index(name)


def decode(index):
    return all_classes[index]


for line in stored_data:
    line_data = line.strip().split(',')
    try:
        classes[line_data[0]] += 1
    except:
        classes[line_data[0]] = 1
    pre_train_x.append([eval(temp) if temp != '?' else -1 for temp in line_data[1:]])
    pre_train_y.append(line_data[0])
# print(pre_train_x)
all_classes = [keys for keys in classes]

for i in range(pre_train_x.__len__()):
    if i % 20 != 0:
        train_x.append(pre_train_x[i])
        train_y.append(pre_train_y[i])
train_x = torch.Tensor(train_x)
train_y = torch.LongTensor([encode(temp) for temp in train_y])

INPUT_DIM = train_x[0].__len__()
OUT_DIM = all_classes.__len__()
NUM_HIDDEN1 = 300
NUM_HIDDEN2 = 100
model = torch.nn.Sequential(
    torch.nn.Linear(INPUT_DIM, NUM_HIDDEN1),
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_HIDDEN1, NUM_HIDDEN2),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(NUM_HIDDEN2, OUT_DIM)

)

if torch.cuda.is_available():
    model = model.cuda()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

BATCH_SIZE = 16
for epoch in range(1000):
    for start in range(0, len(train_x), BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = train_x[start: end]
        batchY = train_y[start: end]
        if torch.cuda.is_available():
            batchX = batchX.cuda()
            batchY = batchY.cuda()
        y_pred = model(batchX)
        loss = loss_fn(y_pred, batchY)
        print("Epoch", epoch, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# random.seed(0)
# test_index = [random.randint(0, pre_train_y.__len__()) for _ in range(10)]
test_x = [pre_train_x[20 * temp] for temp in range(16)]
test_x = torch.Tensor(test_x)
# test_y = [pre_train_y[temp] for temp in test_index]
if torch.cuda.is_available():
    test_x = test_x.cuda()
with torch.no_grad():
    pre_y = model(test_x)

pre_y = pre_y.max(1)[1].cpu().tolist()
predictions = [decode(temp) for temp in pre_y]
for i in range(15):
    print(stored_data[i * 20], predictions[i])
# sum = in range(1, 101):
#     if test_Y[i - 1] == fizz_buzz_encode(i):
#         sum += 1
# print(sum) 0
