

# tqdm usage



```python
   with tqdm(total=len(trainloader)) as pbar:
        pbar.set_description("Train epoch:%d"%epoch)
        for batch_idx, (images, labels) in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            images, labels = images.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            #forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            accuracy = correct*100./total            

            postfix_str = "loss:{:.3f}, accuracy:{:.3f}".format(train_loss/total, accuracy)
            pbar.set_postfix_str(postfix_str)
            pbar.update()
```



```bash
Train epoch:0: 100%|█████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:45<00:00,  8.57it/s, loss:0.023, accuracy:15.834]
test epoch:0: 100%|██████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:03<00:00, 33.31it/s, loss:0.020, accuracy:22.440]
Accuracy of the network on the test images: 22.4 %
saving...
Train epoch:1: 100%|█████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:49<00:00,  7.92it/s, loss:0.016, accuracy:23.978]
test epoch:1: 100%|██████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:03<00:00, 32.58it/s, loss:0.019, accuracy:25.870]
Accuracy of the network on the test images: 25.9 %
saving...
Train epoch:2: 100%|█████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:48<00:00,  8.06it/s, loss:0.014, accuracy:31.814]
test epoch:2: 100%|██████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:03<00:00, 32.33it/s, loss:0.016, accuracy:41.060]
Accuracy of the network on the test images: 41.1 %
saving...
Train epoch:3:  16%|██████████████                                                                        | 64/391 [00:08<00:39,  8.34it/s, loss:0.013, ac^Cracy:38.953]


```







## tensorwatch

```bash
history
import tensorwatch as tw
import torchvision.models
alexnet_model = torchvision.models.alexnet()
graph = tw.draw_model(alexnet_model, [1,3,224,224])
graph.save('./alexnet_model.pdf')
tw.model_stats(alexnet_model)
tw.model_stats(alexnet_model, [1,3,224,224])
model = torchvision.models.resnet.resnet18(pretrained=False)
print(model)
tw.model_stats(resnet, [1,3,224,224])
history
tw.model_stats(model, [1,3,224,224])
```

