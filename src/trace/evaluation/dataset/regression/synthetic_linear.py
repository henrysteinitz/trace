

class SyntheticLinearRegressionDataset(Dataset):

    def __init__(self, A, b, size):
        self.X = [torch.rand_like(b) for _ in range(size)]
        self.Y = [torch.matmul(A, x) for x in self.X]


    def __len__(self):
        return len(self.X)


    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]