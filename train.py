import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import logging

import torch.optim.adam
from preprocess import *
from model import Advanced_CNN_Attention_Model,CNN_attention_model_S,CNN_attention_res_model
import sklearn
from sklearn.metrics import accuracy_score

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='Advanced_CNN_Attention', help="Name of model")
    parser.add_argument('--epochs', type=int, default=400, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate for the optimizer ")
    parser.add_argument('--momentum', type=float, default=0.99, help="momentum for SGD optimizer")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--use_gpu', type=bool, default=True, help="Flag to use GPU if available")
    parser.add_argument('--depth', type=int, default=2, help="Depth of model")
    parser.add_argument('--inter_channel', type=int, default=16, help="Channel of intermediate layers") 
    parser.add_argument('--train_data_path', type=str, default='./fer2013.csv', help="Train dataset path")
    parser.add_argument('--validation_data_path', type=str, default='./fer2013.csv', help="Validation dataset path")
    parser.add_argument('--test_data_path', type=str, default='./Data/test', help="Test dataset path")
    parser.add_argument('--ckp_dir', type=str, default='./ckp', help="Path to checkpoint directory")
    parser.add_argument('--log_path', type=str, default='./log', help="Path to log file")
    parser.add_argument('--if_pretrain', type=bool, default=False, help="Flag to use pre-trained model")
    parser.add_argument('--ckp_path', type=str, default= './ckp/VGG_attention_1_best/VGG_attention_1_acc_0.66.pth', help="relative path to pretrain ckp")

    args = parser.parse_args()

    # Set device
    device = 'cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'

    # Configure the logging to write to a file
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(os.path.join(args.ckp_dir,f'{args.model}_epoch'), exist_ok=True)
    os.makedirs(os.path.join(args.ckp_dir,f'{args.model}_best'), exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.log_path, f'{args.model}.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logging.info(f"Using device: {device}")
    


    # Load data
    # train_data_path = os.path.join(args.train_data_path, "_annotations.csv")
    # validation_data_path = os.path.join(args.validation_data_path, "_annotations.csv")
    # test_data_path = os.path.join(args.test_data_path, "_annotations.csv")
    train_loader = FER_preprocess(args.train_data_path,"Training", batch_size=args.batch_size)
    valid_loader = FER_preprocess(args.train_data_path, "PublicTest",batch_size=args.batch_size)


    # Initialize model
    model = Advanced_CNN_Attention_Model(1,7).to(device)  # Move model to device


    # # Load checkpoint TODO
    if args.if_pretrain:
        checkpoint = torch.load(args.ckp_path)
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # start_epoch = checkpoint['epoch'] + 1
        # loss = checkpoint['loss']

    # loss function, and optimizer  
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,weight_decay=1e-3)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-4)

    # Train loop
    best_acc = 0
    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch+1}/{args.epochs}")

        model.train()  
        total_loss = 0
        total_samples = 0   
        total_correct = 0

        for imgs, labels in train_loader:
            
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(imgs)
            
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            
            optimizer.step()
            
            predictions = model.predict(outputs).long()
            total_correct += (predictions == labels).sum().item()
            total_loss += loss.item()
            total_samples += labels.size(0)

        # Calculate average loss
        train_accuracy = total_correct / total_samples
        average_loss = total_loss / total_samples
        logging.info(f"train accuracy after Epoch {epoch+1}: {train_accuracy:.4f}")
        logging.info(f"Epoch {epoch+1} - train Loss: {average_loss:.4f}")

        if epoch % 50 == 0:
            torch.save({
                # 'epoch': epoch,
                'model_state_dict': model.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
                # 'loss': loss,
            }, os.path.join(args.ckp_dir,f'{args.model}_epoch',f"{args.model}_epoch_{epoch}.pth") )


        model.eval()  
        with torch.no_grad():
            val_loss = 0
            val_sample = 0
            total_correct = 0
            total_samples = 0
    
            for imgs, labels in valid_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss += loss_fn(outputs, labels).item()
                val_sample += args.batch_size
                predictions = model.predict(outputs).long()
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
            
            accuracy = total_correct / total_samples
            avg_val_loss = val_loss / val_sample
            
            if accuracy > best_acc:
                best_acc = accuracy
                torch.save({
                    # 'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                    # 'loss': loss,
                }, os.path.join(args.ckp_dir,f'{args.model}_best',f"{args.model}_acc_{best_acc:.2f}.pth") )
            logging.info(f"Validation Loss after Epoch {epoch+1}: {avg_val_loss:.4f}")
            logging.info(f"Validation accuracy after Epoch {epoch+1}: {accuracy:.4f}")
            

    logging.info("Training completed successfully!")
    
    

if __name__ == "__main__":
    main()

