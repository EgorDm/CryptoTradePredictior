from sklearn.metrics import confusion_matrix


def confusion_matrix_stats(Y_real, Y_predicted):
    tn, fp, fn, tp = confusion_matrix(Y_real, Y_predicted).ravel()

    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    F1 = float(2 * precision * recall) / (precision + recall)
    return precision, recall, F1
