import torch
from fast_pytorch_kmeans import KMeans as KMeans1
import torchvision
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
import numpy as np

class NEW_Strategy:
    def __init__(self, images, net):
        self.images = images
        self.net = net

    def euclidean_dist(self,x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        return dist

    def query(self, n):
        embeddings = self.get_embeddings(self.images)

        index = torch.arange(len(embeddings),device='cuda')

        kmeans = KMeans1(n_clusters=n, mode='euclidean')
        labels = kmeans.fit_predict(embeddings)
        centers = kmeans.centroids

        dist_matrix = self.euclidean_dist(centers, embeddings)
      
        q_idxs = index[torch.argmin(dist_matrix,dim=1)]
 
        return q_idxs,labels
    
    def get_embeddings(self, images):
        embed=self.net.embed
        with torch.no_grad():
            features = embed(images).detach()
        return features

class Cluster_Strategy:
    def __init__(self, images, net, image_syn):
        self.images = images
        self.net = net
        self.image_syn = image_syn

    def euclidean_dist(self,x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def query(self):
        embeddings = self.get_embeddings(self.images)
        imagesyn_embeddings = self.get_embeddings(self.image_syn)
        index = torch.arange(len(embeddings),device='cuda')
        centers = imagesyn_embeddings

        dist_matrix = self.euclidean_dist(centers, embeddings)
      
        q_idxs = index[torch.argmin(dist_matrix,dim=0)]
       
        return q_idxs
    
    def get_embeddings(self, images):
        embed=self.net.embed
        with torch.no_grad():
            features = embed(images).detach()
        return features
    

def cluster_loss(config, dataloader, net, args, n_clusters = 10, random_state = 0):
    
    model = net.to(args.device)

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    features = []
    labels = []

    if "imagenet" in args.dataset:
        class_map = {x: i for i, x in enumerate(config.img_net_classes)}

    for batch_data, batch_labels in dataloader:
        if "imagenet" in args.dataset:
            batch_labels = torch.tensor([class_map[x.item()] for x in batch_labels]).to(args.device)
        features.append(batch_data)
        labels.append(batch_labels)

    """
    After clustering the corresponding data set, 
    you should store the corresponding labels to ensure that the clustering remains consistent in subsequent tests
    """
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=-1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = (features.view(features.size(0), -1))
    clusters = kmeans.fit_predict(features.detach().numpy())

    # Test accuracy separately in each cluster
    cluster_precisions = []
    for i in range(n_clusters):
        cluster_indices = torch.nonzero(torch.from_numpy(clusters) == i).squeeze()
        cluster_data = features[cluster_indices]
        cluster_labels = (torch.tensor(labels)[cluster_indices]).to(device)
  
        if args.dataset == 'CIFAR10' or args.dataset == "svhn":
            cluster_data = cluster_data.view(-1, 3, 32, 32)
            
        elif args.dataset.startswith("imagenet"):
            cluster_data = cluster_data.view(-1, 3, 128, 128)

        cluster_data = cluster_data.to(device)
        
        outputs = model(cluster_data)
     
        _, predicted = torch.max(outputs, 1)
        precision = (predicted == cluster_labels).sum().item() / cluster_labels.size(0)
        cluster_precisions.append(precision)
        mean = np.mean(cluster_precisions)
        min = np.min(cluster_precisions)
    
    return mean, min
