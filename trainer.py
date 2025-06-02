import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from merlion.evaluate.anomaly import ScoreType
from models import ad_predict
from models.reasonable_metric import tsad_reasonable
from models.reasonable_metric import reasonable_accumulator
from models.AOC.trainer.early_stopping import EarlyStopping
import matplotlib.pyplot as plt


sys.path.append("../../COCA")
def Trainer(model, model_optimizer, train_dl, val_dl, test_dl, device, config, idx, seeloss):
    # Start training
    print("Training started ....")

    save_path = "./best_network/" + config.dataset
    os.makedirs(save_path, exist_ok=True)
    early_stopping = EarlyStopping(save_path, idx, patience=100)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    all_epoch_train_loss, all_epoch_test_loss = [], []
    all_sim_qc, all_sim_qprimec, all_sim_qqprime = [],[],[]
    all_sim_qac, all_sim_qa_primec = [],[]
    center = torch.zeros(config.project_channels, device=device)
    center = center_c(train_dl, model, device, center, config, eps=config.center_eps)
    length = torch.tensor(0, device=device)  # radius R initialized with 0 by default.

    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_target, train_score, train_loss, length, total_sim_qc, total_sim_qprimec, total_sim_qqprime, total_sim_qac, total_sim_qa_primec = model_train(model, model_optimizer, train_dl, center,
                                                                    length, config, device, epoch)
        val_target, val_score_origin, val_loss, all_projection = model_evaluate(model, val_dl, center, length, config,
                                                                            device, epoch)
        test_target, test_score_origin, test_loss, all_projection = model_evaluate(model, test_dl, center, length,
                                                                                   config, device, epoch)
        if epoch < config.change_center_epoch:
            center = center_c(train_dl, model, device, center, config, eps=config.center_eps)
        scheduler.step(train_loss)
        if epoch % 1 == 0:
            print(f'\nEpoch : {epoch}\n'
                         f'Train Loss     : {train_loss:.4f}\t | \n'
                         f'Valid Loss     : {val_loss:.4f}\t  | \n'
                         f'Test Loss     : {test_loss:.4f}\t  | \n'
                         )
        all_epoch_train_loss.append(train_loss.item())
        all_epoch_test_loss.append(val_loss.item())
        all_sim_qc.append(total_sim_qc.item())
        all_sim_qprimec.append(total_sim_qprimec.item())
        all_sim_qqprime.append(total_sim_qqprime.item())
        # print("Here's total_sim_qa_primec")
        # print(total_sim_qa_primec)
        all_sim_qac.append(total_sim_qac.item())
        all_sim_qa_primec.append(total_sim_qa_primec.item())
        # print("Here's all_sim_qa_primec")
        # print(all_sim_qa_primec)

        if config.dataset == 'UCR':
            # val_affiliation, val_score, _, _, _ = ad_predict(val_target, val_score_origin, config.threshold_determine,
            #                                            config.detect_nu)
            test_affiliation, test_score, _, _, predict = ad_predict(test_target, test_score_origin, config.threshold_determine,
                                                       config.detect_nu)
            score_reasonable = tsad_reasonable(test_target, predict, config.time_step)
            indicator = test_score.f1(ScoreType.RevisedPointAdjusted)
            early_stopping(score_reasonable, test_affiliation, test_score, indicator, val_score_origin,
                           test_score_origin, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        elif config.dataset == 'SWaT' or config.dataset == 'WADI':
            val_affiliation, val_score, _, _, predict = ad_predict(val_target, val_score_origin, config.threshold_determine,
                                                       config.detect_nu)
            # test_affiliation, test_score, _, _, predict = ad_predict(test_target, test_score_origin, config.threshold_determine,
            #                                            config.detect_nu)
            score_reasonable = tsad_reasonable(test_target, predict, config.time_step)
            indicator = val_score.f1(ScoreType.RevisedPointAdjusted)
            early_stopping(score_reasonable, val_affiliation, val_score, indicator, val_score_origin,
                           test_score_origin, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        elif config.dataset == 'SWaT1' or config.dataset == 'WADI1':
            early_stopping(0, 0, 0, -val_loss.item(), val_score_origin, test_score_origin, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    print("\n################## Training is Done! #########################")
    # according to scores to create predicting labels
    if config.dataset == 'UCR':
        score_reasonable = early_stopping.best_score_reasonable
        # The UCR validation set has no anomaly, so it does not print.
        test_score_origin = early_stopping.best_predict2
        test_affiliation, test_rpa_score, test_pa_score, test_pw_score, predict = ad_predict(test_target,
                                                                                             test_score_origin,
                                                                                             config.threshold_determine,
                                                                                             config.detect_nu)

    elif config.dataset == 'SWaT' or config.dataset == 'WADI':
        val_score_origin = early_stopping.best_predict1
        test_score_origin = early_stopping.best_predict2
        print('best RPA F1: {:.4f}'.format(early_stopping.best_indicator))
        val_affiliation, val_score, _, _, _ = ad_predict(val_target, val_score_origin, config.threshold_determine,
                                                         config.detect_nu)
        test_affiliation, test_rpa_score, test_pa_score, test_pw_score, predict = ad_predict(test_target,
                                                                                             test_score_origin,
                                                                                             config.threshold_determine,
                                                                                             config.detect_nu)
        score_reasonable = reasonable_accumulator(1, 0)
        val_f1 = val_score.f1(ScoreType.RevisedPointAdjusted)
        val_precision = val_score.precision(ScoreType.RevisedPointAdjusted)
        val_recall = val_score.recall(ScoreType.RevisedPointAdjusted)
        print("Valid affiliation-metrics")
        print(
            f'Test precision: {val_affiliation["precision"]:2.4f}  | \tTest recall: {val_affiliation["recall"]:2.4f}\n')
        print("Valid RAP F1")
        print(
            f'Valid F1: {val_f1:2.4f}  | \tValid precision: {val_precision:2.4f}  | \tValid recall: {val_recall:2.4f}\n')


    else:
        val_affiliation, val_score, _, _, _ = ad_predict(val_target, val_score_origin, config.threshold_determine,
                                                         config.detect_nu)
        test_affiliation, test_rpa_score, test_pa_score, test_pw_score, predict = ad_predict(test_target,
                                                                                             test_score_origin,
                                                                                             config.threshold_determine,
                                                                                             config.detect_nu)
        score_reasonable = reasonable_accumulator(1, 0)
        val_f1 = val_score.f1(ScoreType.RevisedPointAdjusted)
        val_precision = val_score.precision(ScoreType.RevisedPointAdjusted)
        val_recall = val_score.recall(ScoreType.RevisedPointAdjusted)
        print("Valid affiliation-metrics")
        print(
            f'Test precision: {val_affiliation["precision"]:2.4f}  | \tTest recall: {val_affiliation["recall"]:2.4f}\n')
        print("Valid RAP F1")
        print(
            f'Valid F1: {val_f1:2.4f}  | \tValid precision: {val_precision:2.4f}  | \tValid recall: {val_recall:2.4f}\n')


    print("Test affiliation-metrics")
    print(
        f'Test precision: {test_affiliation["precision"]:2.4f}  | \tTest recall: {test_affiliation["recall"]:2.4f}\n')
    test_rpa_f1 = test_rpa_score.f1(ScoreType.RevisedPointAdjusted)
    test_rpa_precision = test_rpa_score.precision(ScoreType.RevisedPointAdjusted)
    test_rpa_recall = test_rpa_score.recall(ScoreType.RevisedPointAdjusted)
    print("Test RAP F1")
    print(f'Test F1: {test_rpa_f1:2.4f}  | \tTest precision: {test_rpa_precision:2.4f}  | \tTest recall: {test_rpa_recall:2.4f}\n')

    test_pa_f1 = test_pa_score.f1(ScoreType.PointAdjusted)
    test_pa_precision = test_pa_score.precision(ScoreType.PointAdjusted)
    test_pa_recall = test_pa_score.recall(ScoreType.PointAdjusted)
    print("Test PA F1")
    print(
        f'Test F1: {test_pa_f1:2.4f}  | \tTest precision: {test_pa_precision:2.4f}  | \tTest recall: {test_pa_recall:2.4f}\n')

    test_pw_f1 = test_pw_score.f1(ScoreType.PointAdjusted)
    test_pw_precision = test_pw_score.precision(ScoreType.PointAdjusted)
    test_pw_recall = test_pw_score.recall(ScoreType.PointAdjusted)
    print("Test PW F1")
    print(
        f'Test F1: {test_pw_f1:2.4f}  | \tTest precision: {test_pw_precision:2.4f}  | \tTest recall: {test_pw_recall:2.4f}\n')

    if seeloss:
        epochs = np.arange(1, len(all_epoch_train_loss) + 1)
        max_length = len(all_epoch_train_loss)  # 以 all_epoch_train_loss 的长度为基准
        if len(all_sim_qac) < max_length:
            all_sim_qac = [0] * (max_length - len(all_sim_qac)) + all_sim_qac  # 在前面补 0
        if len(all_sim_qa_primec) < max_length:
            all_sim_qa_primec = [0] * (max_length - len(all_sim_qa_primec)) + all_sim_qa_primec  # 在前面补 0
        loss_data = np.column_stack((epochs, all_epoch_train_loss, all_epoch_test_loss, all_sim_qc, all_sim_qprimec, all_sim_qqprime, all_sim_qac, all_sim_qa_primec))
        np.savetxt(f'experiments_logs/loss-{config.dataset}-{idx}.csv', loss_data, delimiter=',', header='Epoch,TrainLoss,TestLoss,Sim-qc,Sim-qprimec, Sim-qqprime, Sim-qac, Sime-qaprimec', comments='', fmt='%.6f')
        
        fig, ax1 = plt.subplots()

        color1 = '#1f77b4'  # 蓝色
        color2 = '#ff7f0e'  # 橙色
        color3 = '#2ca02c'  # 绿色
        color4 = '#d62728'  # 红色
        color5 = '#9467bd'  # 紫色
        color6 = '#FFD700'  # 金色
        color7 = '#FF69B4'  # 热情粉色
        # ax1.plot(epochs, all_epoch_train_loss, label='Train Loss', color=color1, marker='o', linestyle='-', linewidth=2)
        # ax1.plot(epochs, all_epoch_test_loss, label='Test Loss', color=color2, marker='x', linestyle='--', linewidth=2)
        ax1.plot(epochs, all_epoch_train_loss, label='Train Loss', color=color1, marker='o', linestyle='-', linewidth=2)
        ax1.plot(epochs, all_epoch_test_loss, label='Test Loss', color=color2, marker='x', linestyle='--', linewidth=2)


        ax2 = ax1.twinx()
        ax2.plot(epochs, all_sim_qc, label=r"Sim($q_n, c$)", color=color3, linestyle='-', linewidth=2)
        ax2.plot(epochs, all_sim_qprimec, label=r"Sim($q^{'}_n, c$)", color=color4, linestyle='-.', linewidth=2)
        ax2.plot(epochs, all_sim_qqprime, label=r"Sim($q_n,q^{'}_n$)", color=color5, linestyle=':', linewidth=2)
        ax2.plot(epochs, all_sim_qac, label=r"Sim($q_a, c$)", color=color6, linestyle='-', linewidth=2)
        ax2.plot(epochs, all_sim_qa_primec, label=r"Sim($q^{'}_a, c$)", color=color7, linestyle='--', linewidth=2)
        # 添加标签和标题
        ax1.set_xlabel('Epoch')
        # ax1.set_ylabel('损失', color='black')
        ax1.set_ylabel('Loss', color='black')
        ax1.tick_params(axis='y', labelcolor='black')

        # ax2.set_ylabel('相似度', color='black')
        ax2.set_ylabel('Similarity', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        
        
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        # ax2.legend(loc='upper left', bbox_to_anchor=(1.25, 1)) 

        # 设置图形标题
        plt.title('Loss and Similarity over Epochs')
        # plt.title('损失与相似度随训练轮次变化')
        plt.subplots_adjust(right=0.85)  # 调整右侧留白以容纳图例
        # 显示图形
        # plt.show()
        plt.savefig(f'experiments_logs/loss-{config.dataset}-{idx}.pdf', format='pdf')
        

    return test_score_origin, test_affiliation, test_rpa_score, test_pa_score, test_pw_score, score_reasonable, predict


def model_train(model, model_optimizer, train_loader, center, length, config, device, epoch):

    total_loss, total_f1, total_precision, total_recall = [], [], [], []
    all_target, all_predict = [], []
    total_sim_qc, total_sim_qprimec, total_sim_qqprime = [],[],[]
    total_sim_qac, total_sim_qa_primec = [],[]
    model.train()
    # torch.autograd.set_detect_anomaly(True)
    for batch_idx, (data, target, aug1, aug2) in enumerate(train_loader):
        # send to device
        data, target = data.float().to(device), target.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
        all_data = torch.cat((data, aug1, aug2), dim=0)
        # optimizer
        model_optimizer.zero_grad()
        feature1, feature_dec1 = model(all_data)
        loss_oc, loss_sigam, loss_n, loss_a, dis1, dis_dec1, dis_qq, dis_qac, dis_qaprimec = train(feature1, feature_dec1, center, length, epoch, config, device)
        #把sim传出来
        total_sim_qc.append(dis1)
        total_sim_qprimec.append(dis_dec1)
        total_sim_qqprime.append(dis_qq)
        total_sim_qac.append(dis_qac) if dis_qac is not None else None
        total_sim_qa_primec.append(dis_qaprimec) if dis_qaprimec is not None else None


        loss = loss_oc + config.omega * loss_sigam
        # Update hypersphere radius R on mini-batch distances
        if (config.train_method == 'soft_boundary') and (epoch >= config.freeze_length_epoch):
            length = torch.tensor(get_radius(loss_n, config.nu), device=device)
        score = loss_n
        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()

        target = target.reshape(-1)
        predict = score.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        all_target.extend(target)
        all_predict.extend(predict)

    total_loss = torch.tensor(total_loss).mean()

    total_sim_qac = [tensor for tensor in total_sim_qac if not torch.isnan(tensor).any() and not torch.isinf(tensor).any()]
    total_sim_qa_primec = [tensor for tensor in total_sim_qa_primec if not torch.isnan(tensor).any() and not torch.isinf(tensor).any()]

    return all_target, all_predict, total_loss, length, torch.tensor(total_sim_qc).mean(), torch.tensor(total_sim_qprimec).mean(), torch.tensor(total_sim_qqprime).mean(), torch.tensor(total_sim_qac).mean(), torch.tensor(total_sim_qa_primec).mean()


def model_evaluate(model, test_dl, center, length, config, device, epoch):
    model.eval()
    total_loss, total_f1, total_precision, total_recall = [], [], [], []
    all_target, all_predict = [], []
    all_projection = []
    with torch.no_grad():
        for data, target, aug1, aug2 in test_dl:
            data, target = data.float().to(device), target.long().to(device)
            feature1, feature_dec1 = model(data)
            loss_oc, loss_sigam, loss_n, loss_a, _, _, _, _, _ = train(feature1, feature_dec1, center, length, epoch, config, device)
            score = loss_n
            loss = loss_oc + config.omega * loss_sigam
            if not torch.isnan(loss):
                total_loss.append(loss.item())
            predict = score.detach().cpu().numpy()
            target = target.reshape(-1)
            all_target.extend(target.detach().cpu().numpy())
            all_predict.extend(predict)
            all_projection.append(feature1)
    # average loss
    total_loss = torch.tensor(total_loss).mean()
    all_projection = torch.cat(all_projection, dim=0)
    all_target = np.array(all_target)

    return all_target, all_predict, total_loss, all_projection

def train(feature1, feature_dec1, center, length, epoch, config, device):
    # normalize feature vectors
    center = center.unsqueeze(0)
    center = F.normalize(center, dim=1)
    feature1 = F.normalize(feature1, dim=1)
    feature_dec1 = F.normalize(feature_dec1, dim=1)

    distance1 = F.cosine_similarity(feature1, center, eps=1e-6)

    distance_qc = distance1
    distance_dec1 = F.cosine_similarity(feature_dec1, center, eps=1e-6)

    distance_qprimec = distance_dec1
    distance1 = 1 - distance1
    distance_dec1 = 1 - distance_dec1

    distance_qq = F.cosine_similarity(feature_dec1, feature1, eps=1e-6)

    # Prevent model collapse
    sigma_aug1 = torch.sqrt(feature1.var([0]) + 0.0001)
    sigma_aug2 = torch.sqrt(feature_dec1.var([0]) + 0.0001)
    sigma_loss1 = torch.max(torch.zeros_like(sigma_aug1), (1 - sigma_aug1))
    sigma_loss2 = torch.max(torch.zeros_like(sigma_aug2), (1 - sigma_aug2))
    loss_sigam = torch.mean((sigma_loss1 + sigma_loss2) / 2)

    loss_n = distance1 + distance_dec1
    loss_a = 4 - loss_n

    # The Loss function
    # score = loss_n - loss_a
    score = loss_n
    q_anomaly_c = None
    q_anomaly_c_prime = None

    if epoch < config.warmup:
        loss_oc = torch.mean(loss_n)
    else:
        if config.train_method == 'soft_boundary':
            diff1 = loss_n - length
            loss_oc = length + (1 / config.phi) * torch.mean(torch.max(torch.zeros_like(diff1), diff1))
        elif config.train_method == 'loe_ts':
            # print(int(score.shape[0] * (1 - config.nu)))
            # print(int(score.shape[0] * config.nu))
            _, idx_n = torch.topk(score, int(score.shape[0] * (1 - config.nu)), largest=False,
                                  sorted=False)
            _, idx_a = torch.topk(score, int(score.shape[0] * config.nu), largest=True,
                                  sorted=False)
            # 少一个样本没算计算loss,无伤大雅
            loss_oc = torch.cat([loss_n[idx_n], config.mu * loss_a[idx_a]], 0)
            loss_oc = loss_oc.mean()
            q_anomaly_c = (1 - distance1)[idx_a].mean()
            q_anomaly_c_prime = (1-distance_dec1)[idx_a].mean()
        elif config.train_method == 'loe_soft':
            _, idx_n = torch.topk(score, int(score.shape[0] * (1 - config.nu)), largest=False, sorted=False)
            _, idx_a = torch.topk(score, int(score.shape[0] * config.nu), largest=True, sorted=False)
            loss_oc = torch.cat([loss_n[idx_n], 0.5 * loss_n[idx_a] + 0.5 * loss_a[idx_a]], 0)
            loss_oc = loss_oc.mean()
        elif config.train_method == 'refine':
            _, idx_n = torch.topk(loss_n, int(loss_n.shape[0] * (1 - config.nu)), largest=False,
                                  sorted=False)
            loss_oc = loss_n[idx_n]
            loss_oc = loss_oc.mean()
        else:
            loss_oc = torch.mean(loss_n)

    return loss_oc, loss_sigam, loss_n, loss_a, distance_qc.mean(), distance_qprimec.mean(), distance_qq.mean(), q_anomaly_c, q_anomaly_c_prime

def center_c(train_loader, model, device, center, config, eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    c = center
    model.eval()
    with torch.no_grad():
        for data in train_loader:
            # get the inputs of the batch
            data, target, aug1, aug2 = data
            data = data.float().to(device)
            aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
            all_data = torch.cat((data, aug1, aug2), dim=0)
            outputs, dec = model(all_data)
            n_samples += outputs.shape[0]
            all_feature = torch.cat((outputs, dec), dim=0)
            # all_feature = outputs
            c += torch.sum(all_feature, dim=0)
    c /= (2 * n_samples)
    # c /= (n_samples)

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    # return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
    dist = dist.reshape(-1)
    return np.quantile(dist.clone().data.cpu().numpy(), 1 - nu)



