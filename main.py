from config import *
from lib import *
from data.newsgroups20 import Newsgroup
from torch.utils.data import DataLoader
from models.CNNText_Inception import CNNText_inception
from tensorboardX import SummaryWriter
from torch import optim
import torch.nn.parallel
import tqdm
import torch
from dixitool.pytorch.module import functional as F
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#
seed_everything(1995)
# tensorboardX
writer = SummaryWriter()

#gpu
output_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(output_device)
#dataset train,val
#only train now
print("Start loading data")
#variable args comes from config.py
train_ds = Newsgroup(args.data.train_dir,train=True)
test_ds = Newsgroup(args.data.test_dir, train=False)
#dataloader
train_dl = DataLoader(train_ds,batch_size=args.data.dataloader.batch_size,shuffle=True,drop_last=True)
test_dl = DataLoader(test_ds,batch_size=args.data.dataloader.batch_size,shuffle=False)


print("Finish loading data")
net = CNNText_inception(args.model)
net.to(output_device)
net = nn.DataParallel(net, list(range(2)))
criterion = nn.CrossEntropyLoss()

    #optimizer

#loss function CrossEntropyLoss
optimizer = optim.SGD(net.parameters(), lr=args.train.lr , weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True)

best_acc = 0.0
global_step = 0
for epoch in range(args.train.epochs):
    print("Epoch:{}/{}".format(epoch,args.train.epochs))
    #train
    net.train()
    for idx, data in tqdm.tqdm(enumerate(train_dl)):
        inputs, labels = data
        inputs = inputs.to(output_device)
        labels = labels.to(output_device)
        optimizer.zero_grad()
        
        outputs = net(inputs)
        
        loss = criterion(outputs,labels)
        loss.backward()

        optimizer.step()

        writer.add_scalar('train/train_loss',loss.item(),global_step)

        global_step += 1


    #val
    net.eval()
    corrects = 0
    for idx, data in enumerate(test_dl):
        inputs, labels = data
        inputs = inputs.to(output_device)
        labels = labels.to(output_device)
        with torch.no_grad():
            outputs = net(inputs)
            _, preds = torch.max(outputs,dim=1)
            corrects += torch.sum(preds == labels).cpu().item()

    #准确率
    total_acc = float(corrects) / len(test_ds)
    #tensorboardX可视化
    writer.add_scalar('data/val_acc',total_acc,epoch)

    #保存模型参数
    if total_acc>best_acc:
        best_acc = total_acc
        F.save_model('checkpoints', '%.2f' % total_acc, net)
        #保存模型
    
writer.export_scalars_to_json("./all_scalars.json")
writer.close()
    





    

