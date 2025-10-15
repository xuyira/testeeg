 # --- evaluation loop ---
            if epoch % args.logging_iter == 0:
                gen_sig = []
                real_sig = []
                model.eval()
                with torch.no_grad():
                    with model.ema_scope():
                        process = DiffusionProcess(args, model.net,
                                                   (args.input_channels, args.img_resolution, args.img_resolution))
                        for data in tqdm(test_loader):
                            # sample from the model
                            x_img_sampled = process.sampling(sampling_number=data[0].shape[0])
                            # --- convert to time series --
                            x_ts = model.img_to_ts(x_img_sampled)

                            # special case for temperature_rain dataset
                            if args.dataset in ['temperature_rain']:
                                x_ts = torch.clamp(x_ts, 0, 1)

                            gen_sig.append(x_ts.detach().cpu().numpy()) #32,1000,22
                            real_sig.append(data[0].detach().cpu().numpy()) #32,1000,22

                gen_sig = np.vstack(gen_sig)
                real_sig = np.vstack(real_sig)
                scores = evaluate_model_uncond(real_sig, gen_sig, args)
                for key, value in scores.items():
                    logger.log(f'test/{key}', value, epoch)

real_sig,gen_sig = torch.Tensor(real_sig).float(),torch.Tensor(gen_sig).float()
        scores = compute_all_metrics(real_sig, gen_sig, setup_optimizer,
                                     torch.nn.Sigmoid() if args.dataset == 'temperature_rain' else torch.nn.Identity(),
                                     args.device)
        return scores

def compute_all_metrics(x_real, gens, get_optim_func, pred_activation, device):
    pred_step = 10
    cls = []
    pred = []
    marg = []
    for i in range(10):
        clfscore = compute_classification_score(gens, x_real, get_optim_func, device)
        #predscore = compute_predictive_score(x_real, gens, pred_step, get_optim_func, device, pred_activation)
        marginalscore = compute_test_metrics(gens, x_real)['marginal_loss']
        cls.append(clfscore)
        #pred.append(predscore)
        marg.append(marginalscore)
    clfscore, clf_std = np.mean(cls), np.std(cls)
    #predscore, pred_std = np.mean(pred), np.std(pred)
    marginalscore = np.mean(marg)

    res = {'clf_score_mean': clfscore,
           'clf_score_std': clf_std,
           'marginal_score_mean': marginalscore
           }

    return res

def compute_classification_score(x_fake, x_real, get_optim_func, device):
    x_fake = x_fake.detach().cpu()
    x_real = x_real.detach().cpu()
    X = torch.cat([x_fake, x_real], dim=0)
    Y = torch.cat([torch.ones_like(x_fake[:, 0, 0]), torch.zeros_like(x_real[:, 0, 0])], dim=0)

    randperm = torch.randperm(X.shape[0])

    X_train, Y_train = X[randperm[:int(X.shape[0] * 0.8)]], Y[randperm[:int(X.shape[0] * 0.8)]]

    X_test, Y_test = X[randperm[int(X.shape[0] * 0.8):]], Y[randperm[int(X.shape[0] * 0.8):]]
    model = S4Model(d_input=X.shape[-1], d_state=16, d_output=1, d_model=16, n_layers=1,
                    dropout=0.0, seq2seq=False).to(device)
    trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, Y_train), shuffle=True,
                                              batch_size=128)
    testloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, Y_test), batch_size=128)
    optimizer, _ = get_optim_func(model, lr=0.01, weight_decay=0.0, epochs=100)

    pbar = tqdm(range(100))
    for i in range(100):
        for data, label in trainloader:
            optimizer.zero_grad()
            pred, _ = model(data.to(device))

            loss = torch.nn.BCEWithLogitsLoss()(pred.squeeze(-1), label.to(device))
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            test_loss = 0
            for ind, (data, label) in enumerate(testloader):
                pred, _ = model(data.to(device))
                loss = torch.nn.BCEWithLogitsLoss()(pred.squeeze(-1), label.to(device)).detach().cpu()
                test_loss += loss

            pbar.set_description(f'Epoch {i} Test loss: {test_loss / (ind + 1)}')

    return test_loss

def compute_test_metrics(x_fake, x_real):
    res = dict()
    res['marginal_loss'] = HistoLoss(x_real=x_real, n_bins=50, name='marginal_loss')(x_fake).item()

    return res

class HistoLoss(Loss):
    def __init__(self, x_real, n_bins, **kwargs):
        super(HistoLoss, self).__init__(**kwargs)
        self.densities = list()
        self.locs = list()
        self.deltas = list()
        for i in range(x_real.shape[2]):
            x_i = x_real[..., i].reshape(-1, 1)
            d, b = histogram_torch(x_i, n_bins, density=True)
            self.densities.append(nn.Parameter(d).to(x_real.device))
            delta = b[1:2] - b[:1]
            loc = 0.5 * (b[1:] + b[:-1])
            self.locs.append(loc)
            self.deltas.append(delta)

    def compute(self, x_fake):
        loss = list()

        def relu(x):
            return x * (x >= 0.).float()

        for i in range(x_fake.shape[2]):
            loc = self.locs[i].view(1, -1).to(x_fake.device)
            x_i = x_fake[:, :, i].contiguous().view(-1, 1).repeat(1, loc.shape[1])
            dist = torch.abs(x_i - loc)
            counter = (relu(self.deltas[i].to(x_fake.device) / 2. - dist) > 0.).float()
            density = counter.mean(0) / self.deltas[i].to(x_fake.device)
            abs_metric = torch.abs(density - self.densities[i].to(x_fake.device))
            loss.append(torch.mean(abs_metric, 0))
        loss_componentwise = torch.stack(loss)
        return loss_componentwise

from torch import nn
import torch
import torch.optim as optim

import numpy as np
from tqdm.auto import tqdm
from models.testing_models.s4d import S4D, dropout_fn


class S4Model(nn.Module):

    def __init__(
            self,
            d_input,
            d_state,
            d_output=10,
            d_model=256,
            n_layers=4,
            dropout=0.2,
            prenorm=False,
            bidirectional=False,
            seq2seq=False,
            lr=0.001,
            activation=nn.Identity()
    ):
        super().__init__()

        self.prenorm = prenorm
        self.seq2seq = seq2seq

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(
                    d_model=d_model,
                    d_state=d_state,
                    bidirectional=bidirectional,
                    # postact='glu' if glu else None,
                    dropout=dropout,
                    transposed=True,
                    lr=min(0.001, lr)
                )
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout) if dropout > 0 else nn.Identity())

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)
        self.act = activation

    def forward(self, x, aux=None, t=None, **kwargs):
        """
        Input x is shape (B, L, d_input)
        """

        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        if not self.seq2seq:
            x = x.mean(1)
        # Decode the outputs
        x = self.decoder(x)  # (B, L, d_model) -> (B, L, d_output)
        x = self.act(x)
        return x, None

    def default_state(self, *args, **kwargs):
        return [layer.default_state(*args, **kwargs) for layer in self.s4_layers]