# import utils
from dynamic.utils import *
# print("PyTorch has version {}".format(torch.__version__))

from torch_geometric.datasets import CitationFull
from torch.nn import Linear
from torch_geometric.nn import GCNConv


dataset = CitationFull("../datasets/CitationFull", "Cora")
print_dataset(dataset)

data = dataset[0]  # Get the first graph object.
add_mask(data)
print_data(data)


class GCN(torch.nn.Module) :
    def __init__(self) :
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index) :
        # x = F.dropout(x, p=0.5, training=self.training)  # sparsify
        h = self.conv1(x, edge_index)
        h = h.relu()  # h.tanh
        h = self.conv2(h, edge_index)
        h = h.relu()
        h = self.conv3(h, edge_index)
        h = h.relu()  # Final GNN embedding space.

        # Apply a final (linear) classifier.
        out = self.classifier(h)

        return out, h


model = GCN()
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.


def train(data) :
    optimizer.zero_grad()  # Clear gradients.
    out, h = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask],
                     data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.

    accuracy = {}
    # Calculate training accuracy on our four examples
    predicted_classes = torch.argmax(out[data.train_mask], axis=1)  # [0.6, 0.2, 0.7, 0.1] -> 2
    target_classes = data.y[data.train_mask]
    accuracy['train'] = torch.mean(
        torch.where(predicted_classes == target_classes, 1, 0).float())

    # Calculate validation accuracy on the whole graph
    predicted_classes = torch.argmax(out, axis=1)
    target_classes = data.y
    accuracy['val'] = torch.mean(
        torch.where(predicted_classes == target_classes, 1, 0).float())

    return loss, h, accuracy


for epoch in range(500) :
    loss, h, accuracy = train(data)
    # Visualize the node embeddings every 10 epochs
    # if epoch % 10 == 0 :
    #     visualize(h, color=data.y, epoch=epoch, loss=loss, accuracy=accuracy)
    #     time.sleep(0.3)
