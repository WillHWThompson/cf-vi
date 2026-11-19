import torch

import cfvi

torch.manual_seed(0)
x = 4 * torch.rand(1000, 1) - 2
y = torch.distributions.Gamma(2.0, 2.0 / torch.exp(torch.sin(2 * x[:, 0]))).sample()  # shape 2, mean e^sin(2x)

head = cfvi.Regression(in_features=1, hidden_features=32, likelihood=cfvi.Gamma(), num_data=len(x))
optimizer = torch.optim.Adam(head.parameters(), lr=2e-2)
for _ in range(500):
    optimizer.zero_grad()
    loss = head(x).train_loss_fn(y)  # the negative ELBO per data point, in closed form
    loss.backward()
    optimizer.step()

head.double()  # float64 resolves smaller tail probabilities than float32
x_new = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64)
with torch.no_grad():
    predictive = head(x_new).predictive
    print(predictive.mean)  # E(y | x)
    print(predictive.sf(torch.tensor(4.0, dtype=torch.float64)))  # P(y > 4 | x)
