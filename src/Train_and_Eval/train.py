import os
import torch
from torch import nn

from src.Train_and_Eval.eval import evaluate_model


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device, writer, logger,
                start_epoch=0, config=None, save_checkpoint=True):
    """
    训练模型并在验证集上评估。

    Args:
        model (torch.nn.Module): 要训练的神经网络模型。
        train_loader (torch.utils.data.DataLoader): 包含训练数据的 DataLoader。
        test_loader (torch.utils.data.DataLoader): 包含测试数据的 DataLoader。
        criterion (torch.nn.Module): 损失函数。
        optimizer (torch.optim.Optimizer): 优化器。
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器。
        num_epochs (int): 要训练的总轮数。
        device (torch.device): 用于计算的设备（CPU 或 GPU）。
        writer (torch.utils.tensorboard.SummaryWriter): 用于 TensorBoard 可视化的 SummaryWriter 对象。
        logger (logging.Logger): 用于记录输出的日志对象。
        start_epoch (int, optional): 开始训练的轮数。用于恢复中断的训练。默认为 0。
        config (object, optional): 包含训练配置的对象，如早停参数等。
        save_checkpoint (bool, optional): 是否保存模型检查点。默认为 True。

    Returns:
        dict: 训练过程中表现最佳的模型状态字典。
    """
    best_val_accuracy = 0
    best_model = None
    patience_counter = 0
    last_best_epoch = start_epoch

    min_delta = config.min_delta
    patience = config.patience
    save_dir = config.save_dir

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 100 == 49:
                logger.info('Epoch [%d/%d], Step [%d/%d], Loss: %.4f',
                            epoch + 1, num_epochs, i + 1, len(train_loader), loss.item())
        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total

        test_loss, metrix, _, _ = evaluate_model(model, test_loader, criterion, device, logger, class_result=False)
        test_accuracy, aa, kappa = metrix

        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        writer.add_scalar('Loss/val', test_loss, epoch)
        writer.add_scalar('Accuracy/val', test_accuracy, epoch)

        logger.info('Epoch [%d/%d], Train Loss: %.4f, Train Acc: %.4f, Val Loss: %.4f, Val Acc: %.4f',
                    epoch + 1, num_epochs, epoch_loss, epoch_acc, test_loss, test_accuracy)

        if test_accuracy > best_val_accuracy + min_delta:
            best_val_accuracy = test_accuracy
            if isinstance(model, nn.DataParallel):
                best_model = model.module.state_dict()
            else:
                best_model = model.state_dict()
            last_best_epoch = epoch
            patience_counter = 0
            logger.info('New best model saved with validation accuracy: %.4f', best_val_accuracy)
        else:
            patience_counter += 1

        # 检查是否应该早停
        if patience_counter >= patience and config.stop_train:
            logger.info(f'Early stopping triggered. No improvement for {patience} epochs.')
            break

        # 每20个epoch保存一次检查点
        if (epoch + 1) % 20 == 0 and save_checkpoint:
            model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_accuracy': best_val_accuracy,
                'best_model': best_model
            }
            save_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save(checkpoint, save_path)
            logger.info(f'Checkpoint saved at epoch {epoch + 1}: {save_path}')

    logger.info(f'Training completed. Best validation accuracy: {best_val_accuracy:.4f} at epoch {last_best_epoch + 1}')
    return best_model
