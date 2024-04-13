import matplotlib.pyplot as plt
import numpy as np
import csv
def main():
    # ...之前的代码...
    train_loss_list = []
    test_loss_list = []

    # 在训练开始前创建CSV文件并写入表头
    with open('loss_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Test Loss'])
        
    for epoch in range(1,10):
        # ...训练过程的代码...
        
        # 统计平均训练损失
        avg_train_loss=[]
        avg_train_loss.append(50/epoch)

        train_loss_list.append(np.mean(avg_train_loss))

        # ...测试过程的代码...
        
        # 统计平均测试损失
        avg_test_loss = 60/epoch
        test_loss_list.append(avg_test_loss)


        # 每个 epoch 结束后打开CSV文件并追加数据
        with open('loss_data.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_loss_list[-1], test_loss_list[-1]])

        # 绘制损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_list, label='Train Loss')
        plt.plot(test_loss_list, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Testing Loss per Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig('loss_curve_epoch_{}.png'.format(epoch))  # 保存图像文件
        plt.close()  # 关闭图形，避免资源占用

if __name__ == '__main__':
    main()
