import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import torch

def measurement_metrics(model, dataset):
    name = model + '_' + dataset
    name_with_mask = model + '_' + dataset + '_mask'

    # 加载.npy文件
    data1 = np.load('./output/' + name + '/Debug/results/Debug/sv/preds.npy')
    data2 = np.load('./output/' + name + '/Debug/results/Debug/sv/trues.npy')
    print(data1.shape)
    print(data2.shape)

    preds = data1
    trues = data2

    maes = []
    mses = []

    # 遍历每个图像
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            # 计算每个图像的 MAE 和 MSE
            mae = np.mean(np.abs(preds[i, j] - trues[i, j]))
            mse = np.mean((preds[i, j] - trues[i, j])**2)

            maes.append(mae)
            mses.append(mse)

    print("不加mask")
    mae_result1 = np.mean(maes)
    print("所有图像的平均 MAE:", np.mean(maes))
    mse_result1 = np.mean(mses)
    print("所有图像的平均 MSE:", np.mean(mses))



    data1 = np.load('./output/' + name_with_mask + '/Debug/results/Debug/sv/preds.npy')
    data2 = np.load('./output/' + name_with_mask + '/Debug/results/Debug/sv/trues.npy')
    print(data1.shape)
    print(data2.shape)

    preds = data1
    trues = data2

    maes = []
    mses = []

    # 遍历每个图像
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            # 计算每个图像的 MAE 和 MSE
            mae = np.mean(np.abs(preds[i, j] - trues[i, j]))
            mse = np.mean((preds[i, j] - trues[i, j])**2)

            maes.append(mae)
            mses.append(mse)

    print("加mask")
    mae_result2 = np.mean(maes)
    print("所有图像的平均 MAE:", np.mean(maes))
    mse_result2 = np.mean(mses)
    print("所有图像的平均 MSE:", np.mean(mses))



    mae_rate = (mae_result1 - mae_result2) * 100 / mae_result1
    mse_rate = (mse_result1 - mse_result2) * 100 / mse_result1

    return mae_rate, mse_rate

if __name__ == '__main__':
    mae_rate, mse_rate = measurement_metrics('simvp', 'sevir')
