def confusion_matrix(self, num_of_class, pred, true):
    # =============================== EDIT HERE ===============================
    if pred[idx] == true[idx]:
        result = 1
    else:
        result = 0
    
    if class_idx == true[idx]:
        c_m_c[class_idx]['TP'] += result
        c_m_c[class_idx]['FN'] += 1-result
    else:
        if class_idx == pred[idx]:
            c_m_c[class_idx]['FP'] += 1
        else:
            c_m_c[class_idx]['TN'] += 1
        
    # =========================================================================
    return c_m_c

def precision(self, TP, FP, FN, TN):
    out = None
    # =============================== EDIT HERE ===============================
    out = TP / (TP + FP)
    # =========================================================================
    return out

def recall(self, TP, FP, FN, TN):
    out = None
    # =============================== EDIT HERE ===============================
    out = TP/ (TP + FN)
    # =========================================================================
    return out

def f_measure(self, precision, recall, beta=1.0):
    out = None
    # =============================== EDIT HERE ===============================
    out = (beta**2 + 1)*precision*recall/(beta*beta*precision + recall)
    # =========================================================================
    return out