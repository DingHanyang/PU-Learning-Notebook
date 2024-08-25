import torch as t

def EStep(P_yi1_xi, eta, PIndicator):
    P_yi1_xi = P_yi1_xi.squeeze()               # P(yi=1|xi)
    eta = eta.squeeze()                         # eta = P(qi=1|yi=1,xi)
                                                # P(qi|yi=1,xi)    
    P_qi_yi1xi = ((1-eta) ** (1-PIndicator)) * (eta ** (PIndicator))
    P_qi_yi0xi = (1-PIndicator)                 # P(qi|yi=0,xi)
    P_tilte_y1 = P_yi1_xi * P_qi_yi1xi          # P(yi=1,qi|xi)
    P_tilte_y0 = (1-P_yi1_xi) * P_qi_yi0xi      # P(yi=0,qi|xi)
                                                # P(yi,qi|xi)
    P_tilte_y_unn = t.stack([P_tilte_y0, P_tilte_y1], 1) 
                                                # P_tilte(yi) = P(yi=1|xi,qi)
    P_tilte_y = P_tilte_y_unn / P_tilte_y_unn.sum(1).view(-1,1) 
    return P_tilte_y

def MStep(P_yi1_xi, eta, PIndicator, P_tilte_y):
    P_yi1_xi = P_yi1_xi.squeeze()               # P(yi=1|xi)
    eta = eta.squeeze()                         # eta = P(qi=1|yi=1,xi)
                                                # P(qi|yi=1,xi)
                                                # 
                                                #     
    P_qi_yi1xi = ((1-eta) ** (1-PIndicator)) * (eta ** (PIndicator))
    P_qi_yi0xi = (1-PIndicator)                 # P(qi|yi=0,xi)
    P_tilte_y1 = P_yi1_xi * P_qi_yi1xi          # P(yi=1,qi|xi)
    P_tilte_y0 = (1-P_yi1_xi) * P_qi_yi0xi      # P(yi=0,qi|xi)
                                                # P(yi,qi|xi)
    P_tilte_y_unn = t.stack([P_tilte_y0, P_tilte_y1], 1) + 1e-10

    ell = P_tilte_y * P_tilte_y_unn.log()

    return ell.sum(1).mean()
    

def ACCU(P_yi1_xi, y):
    return ((P_yi1_xi > 0.5).float() == y.float()).float().mean()