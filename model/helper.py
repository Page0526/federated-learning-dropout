import torch 
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from simulation.reba import RelaxedBSM

def get_prototypes(net, dataloader, num_classes, device):
    net.eval()
    prototypes = [[] for _ in range(num_classes)]
    with torch.no_grad():
        for im, labels in dataloader:
            im = im.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)
            features, _ = net(im)
            for i, label in enumerate(labels):
                prototypes[label.item()].append(features[i])
    class_prototypes = torch.zeros((num_classes, features.shape[1]), device = device)
    for c in range(num_classes):
        if prototypes[c]:
            class_prototypes[c] = torch.stack(prototypes[c]).mean(dim = 0)
        else:
            class_prototypes[c] = torch.zeros(features.shape[1], device = device)
    return class_prototypes

def feature_augmentation(features, labels, prototypes, num_classes, lam=1.0):
    """Perform feature augmentation by transferring intra-class variance to missing classes."""
    aug_features = []
    aug_labels = []
    for i, label in enumerate(labels):
        src_class = label.item()
        # Cycle through classes for augmentation
        tgt_class = (src_class + 1) % num_classes
        if torch.all(prototypes[tgt_class] == 0):
            continue
        # Compute augmented feature: \tilde{h}_{j,k} = p_j + \lambda (h_{i,k} - p_i)
        aug_feature = prototypes[tgt_class] + lam * (features[i] - prototypes[src_class])
        aug_features.append(aug_feature)
        aug_labels.append(tgt_class)
    
    if aug_features:
        aug_features = torch.stack(aug_features)
        aug_labels = torch.tensor(aug_labels, dtype=torch.long, device=features.device)
        return aug_features, aug_labels
    else:
        return None, None
    
def train(model, dataloader, optimizer, criterion, epochs, device):
    num_classes = 2
    lam = 1.0
    mu = 0.1
    rebafl = RelaxedBSM(dataloader, num_classes=num_classes, device=device)
    prototypes = get_prototypes(model, dataloader, num_classes, device)
    global_prior = rebafl.prior_y(recalculate=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        print(f"Epoch {epoch + 1}/{epochs}")
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            features, logits = model(inputs)
            balanced_probs = rebafl.bsm1(logits)
            loglikelihood = torch.log(balanced_probs + 1e-8)
            loss0 = criterion(loglikelihood, labels)

            aug_features, aug_labels = feature_augmentation(features, labels, prototypes, num_classes, lam)
            loss1 = 0.0
            if aug_features:
                aug_logits = model.classifier(aug_features)
                aug_probs = rebafl.bsm1(aug_logits)
                aug_loglikelihood = torch.log(aug_probs + 1e-8)
                loss1 = criterion(aug_loglikelihood, aug_labels)
            
            loss = loss0 + mu * loss1
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total

        print(f"Train - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    return avg_loss, accuracy


def test(model, dataloader, criterion, device):
  
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            _, logits = model(inputs)

            loss = criterion(logits, labels)

            total_loss += loss.item()
            predicted = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    print(f"Test - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy




def train_model(dataset, model, epochs=10, batch_size=32, learning_rate=0.001):
    """train model on dataset """
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    train(model, train_loader, optimizer, criterion, epochs, device)

    test(model, val_loader, criterion, device)
    test(model, test_loader, criterion, device)

    return model