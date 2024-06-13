import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def show_dataset(x_train):
    img_rows, img_cols = x_train.shape[1], x_train.shape[2]
    col1 = 10
    row1 = 1

    # Show a few sample digits from the training set
    plt.rcParams['figure.figsize'] = (1.0, 1.0) # set default size of plots
    col2 = 20
    row2 = 5
    fig = plt.figure(figsize=(col2, row2))
    for index in range(col1*row1, col1*row1 + col2*row2):
        fig.add_subplot(row2, col2, index - col1*row1 + 1)
        plt.axis('off')
        plt.imshow(x_train[index]) # index of the sample picture
    plt.show()
    

def print_history_chart(history):
    accuracy =history.history['accuracy']
    val_accuracy =history.history['val_accuracy']
    
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']


    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(range(1,len(accuracy)+1),accuracy,color='red',label = "Training Accuracy")
    plt.plot(range(1,len(accuracy)+1),val_accuracy,color='blue',label = "Validation Accuracy")
    plt.ylabel('accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(range(1,len(accuracy)+1),loss,color='red',label = "Training Loss")
    plt.plot(range(1,len(accuracy)+1),val_loss,color='blue',label = "Validation Loss")
    plt.ylabel('Cross Entropy')
    plt.title('Model')
    plt.legend()
    plt.title('Training and Validation Loss')
    

def full_evaluate(model, X, y):
    results = {}
    y_pred = np.argmax(model.predict(X),1)
    results['accuracy_score'] = accuracy_score(y_pred, y)
    print('accuracy_score:', results['accuracy_score'])
    results['f1_score_macro'] = f1_score(y_pred, y, average='macro')
    print('f1_score_macro:', results['f1_score_macro'])
    results['f1_score_weighted'] = f1_score(y_pred, y, average='weighted')
    print('f1_score_weighted:', results['f1_score_weighted'])
    return results


def full_evaluate_differencing(model, X, y):
    results = {}
    y_pred = np.argmax(model.predict(X), 1)
    acc = 0
    c_m = None
    flag = True

    for i in range(1,11):
        true_sim = (y == np.concatenate([y[i:], y[:i]])).reshape(-1)
        pred_sim = (y_pred == np.concatenate([y_pred[i:], y_pred[:i]]))
        acc += accuracy_score(true_sim, pred_sim)
        if flag:
            c_m = confusion_matrix(true_sim, pred_sim)
        else:
            c_m += confusion_matrix(true_sim, pred_sim)

    acc = acc/10
    results['accuracy_score'] = acc
    print('Accuracy:', acc)
    results['confusion_matrix'] = c_m
    print('Confusion matrix:\n', c_m)
    
    def f1(mat):
        tp = mat[1][1] 
        fp = mat[0][1] 
        fn = mat[1][0] 
        tn = mat[0][0] 
        prec, rec = tp/(tp+fp), tp/(tp+fn)
        return 2*prec*rec/(prec+rec)
    
    results['f1_score'] = f1(c_m)
    print('F1-score:', f1(c_m))
    return {}


