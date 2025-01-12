import os

import torch

from src.Train_and_Eval.eval import evaluate_model


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, writer, logger,
                start_epoch=0, config=None):
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
            scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 100 == 49:
                logger.info('Epoch [%d/%d], Step [%d/%d], Loss: %.4f',
                            epoch + 1, num_epochs, i + 1, len(train_loader), loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total

        val_loss, val_accuracy, _, _ = evaluate_model(model, val_loader, criterion, device, logger, class_result=False)

        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        logger.info('Epoch [%d/%d], Train Loss: %.4f, Train Acc: %.4f, Val Loss: %.4f, Val Acc: %.4f',
                    epoch + 1, num_epochs, epoch_loss, epoch_acc, val_loss, val_accuracy)

        if val_accuracy > best_val_accuracy + min_delta:
            best_val_accuracy = val_accuracy
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
        if (epoch + 1) % 20 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
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