class ReplayBuffer(Dataset):
    Memory = namedtuple("Memory", ["state", "action", "next_state", "reward", "done"])

    def __init__(self, capacity=5000):
        self.memory = deque(maxlen=capacity)

    def append(self, state, action, next_state, reward, done):
        self.memory.append(self.Memory(state, action, next_state, reward, done))

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        return self.memory[idx]


class CustomBatchSampler(BatchSampler):
    def __init__(self, data_source, batch_size=32, drop_last=False):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for _ in range(len(self.data_source)):
            idx = random.randint(0, len(self.data_source) - 1)  # Example: random sampling
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size
