
import torch

def prepare_data(data_train_features, 
                data_train_target, 
                data_test_features, 
                data_test_target, 
                data_validation_features, 
                data_validatation_target,
                batch_size ):
    
    data_train = torch.tensor(data_train_features, dtype=torch.float32)
    target_train = torch.tensor(data_train_target, dtype=torch.float32)

    data_test = torch.tensor(data_test_features, dtype=torch.float32)
    target_test = torch.tensor(data_test_target, dtype=torch.float32)

    data_val = torch.tensor(data_validation_features, dtype=torch.float32)
    target_val = torch.tensor(data_validatation_target, dtype=torch.float32)

    print("Input Shapes:")
    print(data_train.shape, data_val.shape, data_test.shape)
    print("Target Shapes:")
    print(target_train.shape, target_val.shape, target_test.shape)

    train_dataset = torch.utils.data.TensorDataset(data_train, target_train)
    test_dataset = torch.utils.data.TensorDataset(data_test, target_test)
    val_dataset = torch.utils.data.TensorDataset(data_val, target_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader