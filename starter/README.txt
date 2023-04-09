TaskA:
Implemented the PointNet architecture in the model.py file.

TaskB:
Implemented the CorrNet architecture in the model.py file.

TaskC:
Implemented the NTcrossentropy loss function in the train.py file.

TaskD:

For train_corrmask = 0 and distance_threshold = 0.01:
Best model according to validation accuracy occurred at Epoch: 94, LR: 0.00100000, losses are as follows:
train_loss: 1025.54915946. train_acc: 0.59913747.
val_loss: 1133.87621307. val_acc: 0.56384957.
test_loss: 1140.05135345. test_acc: 0.52675712.

For train_corrmask = 0 and distance_threshold = 0.02:
Best model according to validation accuracy occurred at Epoch: 97, LR: 0.00100000, losses are as follows:
train_loss: 930.43209185. train_acc: 0.94144024.
val_loss: 1084.11852264. val_acc: 0.92445910.
test_loss: 1179.56261444. test_acc: 0.89423919.

For train_corrmask = 0 and distance_threshold = 0.04:
Best model according to validation accuracy occurred at Epoch: 51, LR: 0.00100000, losses are as follows:
train_loss: 1105.07366616. train_acc: 0.99809322.
val_loss: 1222.20777130. val_acc: 0.99764651.
test_loss: 1316.02962494. test_acc: 0.99124306.

For train_corrmask = 1 and distance_threshold = 0.01:
Best model according to validation accuracy occurred at Epoch: 87, LR: 0.00100000, losses are as follows:
Validation set: Average fitted rotation:  [ 2.16989206 -2.08670042 -0.66719806]
Test set: Average fitted rotation:  [ 2.2429322   1.2680177  -0.95615139]
train_loss: 0.48254553. train_acc: 0.76293430.
val_loss: 0.51418624. val_acc: 0.75582284.
test_loss: 0.50977021. test_acc: 0.76100135.

For train_corrmask = 1 and distance_threshold = 0.02:
Best model according to validation accuracy occurred at Epoch: 100, LR: 0.00100000, losses are as follows:
Validation set: Average fitted rotation:  [ 1.48767515 -6.98372588 -0.76175351]
Test set: Average fitted rotation:  [-0.29842349 -1.70647408  0.6873297 ]
train_loss: 0.52223992. train_acc: 0.74458625.
val_loss: 0.53602837. val_acc: 0.74381924.
test_loss: 0.53404215. test_acc: 0.74739879.


For train_corrmask = 1 and distance_threshold = 0.04:
Best model according to validation accuracy occurred at Epoch: 89, LR: 0.00100000, losses are as follows:
Validation set: Average fitted rotation:  [ 3.33971124 -4.5137773   4.3871357 ]
Test set: Average fitted rotation:  [1.21715224 0.99535642 2.26499862]
train_loss: 0.53974576. train_acc: 0.72567227.
val_loss: 0.56495960. val_acc: 0.72649384.
test_loss: 0.55748304. test_acc: 0.70353669.


TaskE:

For distance_threshold = 0.01:
Best model according to validation accuracy occurred at Epoch: 87, LR: 0.00100000
3D rotation angles for the test set:  [ 2.2429322   1.2680177  -0.95615139]

For distance_threshold = 0.02:
Best model according to validation accuracy occurred at Epoch: 100, LR: 0.00100000
3D rotation angles for the test set:  [-0.29842349 -1.70647408  0.6873297 ]

For distance_threshold = 0.04:
Best model according to validation accuracy occurred at Epoch: 89, LR: 0.00100000
3D rotation angles for the test set:  [1.21715224 0.99535642 2.26499862]
