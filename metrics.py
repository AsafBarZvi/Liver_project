import numpy as np

class lp_metrics():
    def __init__(self):
        self.results = {'dice':0, 'precision':0, 'recall':0, 'accuracy':0}
        self.tp_val = 0.
        self.tn_val = 0.
        self.fp_val = 0.
        self.fn_val = 0.

    def calc_metrics(self, true_labels, pred_labels):
        # True Positive (TP): we predict a label of 1 (positive), and the true label is 1
        self.tp_val += np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
        # True Negative (TN): we predict a label of 0 (negative), and the true label is 0
        self.tn_val += np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
        # False Positive (FP): we predict a label of 1 (positive), but the true label is 0
        self.fp_val += np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
        # False Negative (FN): we predict a label of 0 (negative), but the true label is 1
        self.fn_val += np.sum(np.logical_and(pred_labels == 0, true_labels == 1))

    def summarize_metrics(self):
        #if self.fn_val+self.fp_val+2*self.tp_val == 0:
        #    dice = 0
        #else:
        #    dice = 2*self.tp_val/(self.fn_val+self.fp_val+2*self.tp_val)
        #if self.tp_val+self.fp_val == 0:
        #    precision = 0
        #else:
        #    precision = self.tp_val/(self.tp_val+self.fp_val)
        #if self.tp_val+self.fn_val == 0:
        #    recall = 0
        #else:
        #    recall = self.tp_val/(self.tp_val+self.fn_val)

        dice = 2*self.tp_val/(self.fn_val+self.fp_val+2*self.tp_val)
        precision = self.tp_val/(self.tp_val+self.fp_val)
        recall = self.tp_val/(self.tp_val+self.fn_val)

        accuracy = (self.tp_val + self.tn_val) / (self.tp_val+self.tn_val+self.fp_val+self.fn_val)

        self.results['dice'] = dice
        self.results['precision'] = precision
        self.results['recall'] = recall
        self.results['accuracy'] = accuracy

        return self.results

