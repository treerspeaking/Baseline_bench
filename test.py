from data import Cifar10
from torchvision.transforms import v2

train_data = Cifar10(True, v2.ToTensor())

label_dataloader, unlabeled_dataloader = train_data.get_dataloader(labeled_batch_size=2, labeled_num_worker=4,shuffle = True, unlabeled=True, labeled_size=2000, unlabeled_batch_size=32, unlabeled_num_worker = 4,seed= None)

data_load = zip(label_dataloader, unlabeled_dataloader)
i = 0

for x_u in data_load:
    i += 1
    # print(x_u)
    
print(i)