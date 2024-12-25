import argparse
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import utils
from net import SGRR, Gumble_Softmax, SinkhornDistance
from tqdm import tqdm
from loss import TotalCodingRate, CompressLoss, NcutLoss
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import GradScaler, autocast
from metric import clustering_accuracy
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


parser = argparse.ArgumentParser(description='Unsupervised Learning')
# Model settings
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--feat_dims', type=int, default=128)
parser.add_argument('--hid_dims', type=int, default=4096)
parser.add_argument('--cluster_dims', type=int, default=10)
# Training settings
parser.add_argument('--eps', type=float, default=0.5,
                    help='coding error epsilon')
parser.add_argument('--n_weight', type=float, default=1.0,
                    help='params of Ncut loss')
parser.add_argument('--gamma', type=int, default=100,
                    help='params of orthogonal penalty in Ncut')
parser.add_argument('--nz', type=int, default=100,
                    help='none zeros in batch affinity')
# Optimizer settings
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--wd', type=float, default=5e-4)
parser.add_argument('--bs', type=int, default=512)
parser.add_argument('--epo', type=int, default=50)
parser.add_argument('--w_epo', type=int, default=10)
# Searching parameters
parser.add_argument('--optim', type=str, default='adam',
                    help='sgd or adam')
parser.add_argument('--kernel', dest='kernel', action='store_true')
parser.set_defaults(thres=False)
parser.add_argument('--sigma', type=float, default=1.0,
                    help='scale param of Gaussian kernel')
parser.add_argument('--post', type=str, default='topk',
                    help='topk or ds (doubly stochastic)')
args = parser.parse_args()


def main():
    utils.same_seeds()

    with open('./datasets/cifar10_train.pt', 'rb') as f:
        data = torch.load(f)
        train_data, train_label = data.get("features"), data.get("ys")
        print('train data: {}x{}'.format(train_data.shape[0],train_data.shape[1]))

    with open('./datasets/cifar10_test.pt', 'rb') as f:
        data = torch.load(f)
        test_data, test_label = data.get("features"), data.get("ys")
        print('test data: {}x{}'.format(test_data.shape[0],test_data.shape[1]))

    cluster_num = int(max(train_label)) - int(min(train_label)) + 1

    sgr = SGRR(in_dim=train_data.shape[1], hid_dim=args.hid_dims, feat_dim=args.feat_dims, cluster_dim=cluster_num).to(args.gpu)
    sink_layer = SinkhornDistance(0.175, max_iter=5)

    if args.optim == 'sgd' or args.optim == 'SGD':
        optimizer = optim.SGD(sgr.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=False)
    elif args.optim == 'adam' or args.optim == 'Adam':
        optimizer = optim.Adam(sgr.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        raise AttributeError('Unexpected optimizer!')
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epo, eta_min=0,last_epoch=-1)

    scaler = GradScaler()
    tcr_loss = TotalCodingRate(eps=args.eps)
    compress_loss = CompressLoss(eps=args.eps)
    g_softmax = Gumble_Softmax(tau=1)

    iter_per_epoch = train_data.shape[0] // args.bs
    pbar = tqdm(range(args.epo), ncols=120)
    for epoch in pbar:
        pbar.set_description(f"Epoch {epoch}")
        randidx = torch.randperm(train_data.shape[0])
        sgr.train()

        utils.warmup_lr(optimizer, epoch, args.lr, warmup_epoch=args.w_epo)

        for i in range(iter_per_epoch):
            batch_idx = randidx[i * args.bs : (i + 1) * args.bs]
            batch_data = train_data[batch_idx].to(args.gpu)
            batch_label = train_label[batch_idx]
            with autocast(enabled=True):
                feat, logits = sgr(batch_data)
                feat, logits = F.normalize(feat, dim=1), g_softmax(logits)

                if args.kernel:
                    C = utils.full_affinity(X=feat, sigma=1)
                else:
                    C = sgr.get_aff(batch_data)
                
                if args.post == 'ds':
                    C = sink_layer(C)
                    C = C * C.shape[-1]
                    C = C[0]
                elif args.post == 'topk':
                    zeros = torch.zeros_like(C).to(args.gpu)
                    vals, _ = C.topk(k=args.nz,dim=1)
                    val_min = torch.min(vals,dim=-1).values.unsqueeze(-1).repeat(1,args.bs)
                    ge = torch.ge(C,val_min)
                    C = torch.where(ge,C,zeros)
                    C = F.normalize(C, dim=1)
                else:
                    raise AttributeError('Choose correct post-processing.')

                loss_TCR = tcr_loss(feat)
                loss_compress = compress_loss(feat, logits, cluster_num)
                spectral, orth_reg = NcutLoss(C, logits)
                loss_Ncut = spectral + args.gamma * orth_reg

                if epoch <= args.w_epo:
                    loss = loss_TCR + loss_Ncut
                else:
                    loss = loss_TCR + loss_compress + args.n_weight * loss_Ncut


            with torch.no_grad():
                batch_pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
                batch_acc = clustering_accuracy(batch_label, batch_pred)

                pbar.set_postfix(Lap="{:3.2f}".format(spectral.item()),
                                Orth="{:3.2f}".format(orth_reg.item()),
                                CodingRate="{:3.2f}".format(-loss_TCR.item()),
                                Compateness="{:3.2f}".format(loss_compress.item()),
                                ACC="{:3.3f}".format(batch_acc.item())
                                )
            
            optimizer.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()
        pbar.update(1)


    print('Evaluating on the test set...')
    with torch.no_grad():
        sgr.eval()
        test_data = test_data.float().to(args.gpu)
        _, logits = sgr(test_data)
        
        C = sgr.get_aff(test_data).detach().cpu()
        C = utils.get_sparse_coeff(C,non_zeros=1000)
        Aff = 0.5 * (np.abs(C) + np.abs(C).T)
        sc_preds = utils.spectral_clustering(Aff, cluster_num, cluster_num)
        acc = clustering_accuracy(test_label, sc_preds)
        nmi = normalized_mutual_info_score(test_label, sc_preds, average_method='geometric')
        ari = adjusted_rand_score(test_label, sc_preds)
        print('Spectral Clustering: ACC={:.3f}, NMI={:.3f}, ARI={:.3f}'.format(acc, nmi, ari))
    
        prob = g_softmax(logits)
        preds = torch.argmax(prob.detach().cpu(), dim=1)
        acc = clustering_accuracy(test_label, preds)
        nmi = normalized_mutual_info_score(test_label, preds, average_method='geometric')
        ari = adjusted_rand_score(test_label, preds)
        print('Cluster head: ACC={:.3f}, NMI={:.3f}, ARI={:.3f}'.format(acc, nmi, ari))


if __name__ == '__main__':
    main()



