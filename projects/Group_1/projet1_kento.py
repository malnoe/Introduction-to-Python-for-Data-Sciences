import numpy as np
from functions import Function1, Function2
from baseloss import LossFunction
from baseoptimizer import Optimizer
from utils import make_random_func1, make_random_func2

n= 3
function1 = make_random_func1(n)

def GradientDescent(eps, theta0, L, eta):
    p = -eta*Function1.grad_oracle(L, theta0)
    theta1 = theta0 + p
    liste_theta = [theta0, theta1]
    while (np.linalg.norm(liste_theta[-1] - liste_theta[-2])) > eps :
        p = -eta * Function1.grad_oracle(L, liste_theta[-1])
        liste_theta.append(liste_theta[-1] + p)
        
    print("Nombre d'étapes : "+ str(len(liste_theta)))
    return liste_theta[-1]


def GradientDescent1(eps, theta0, L, eta):
    p = -eta*Function1.grad_oracle(L, theta0)
    theta1 = theta0 + p
    liste_theta = [theta0, theta1]
    while (np.linalg.norm(liste_theta[-1] - liste_theta[-2])) > eps :
        p = -eta * Function1.grad_oracle(L, liste_theta[-1])
        liste_theta.append(liste_theta[-1] + p)
        
    return len(liste_theta)


def NewtonDescentNaive(eps, theta0, L, eta):
    p = -eta*np.dot(np.linalg.inv(Function1.hessian_oracle(L,theta0)), Function1.grad_oracle(L,theta0))
    theta1 = theta0 + p
    liste_theta = [theta0, theta1]
    while (np.linalg.norm(liste_theta[-1] - liste_theta[-2])) > eps :
        p = -eta*np.dot(np.linalg.inv(Function1.hessian_oracle(L,liste_theta[-1])), Function1.grad_oracle(L,liste_theta[-1]))
        liste_theta.append(liste_theta[-1] + p)

    print("Nombre d'étapes : "+ str(len(liste_theta)))
    return liste_theta[-1]


def NewtonDescentNaive1(eps, theta0, L, eta):
    p = -eta*np.dot(np.linalg.inv(Function1.hessian_oracle(L,theta0)), Function1.grad_oracle(L,theta0))
    theta1 = theta0 + p
    liste_theta = [theta0, theta1]
    while (np.linalg.norm(liste_theta[-1] - liste_theta[-2])) > eps :
        p = -eta*np.dot(np.linalg.inv(Function1.hessian_oracle(L,liste_theta[-1])), Function1.grad_oracle(L,liste_theta[-1]))
        liste_theta.append(liste_theta[-1] + p)

    return len(liste_theta)


def NewtonDescentClever(theta0, L):
    return theta0 - np.dot(np.linalg.inv(L.A),L.b)



def BfgsDescent(eps, theta0,L,eta):
    # On definit les variables : n, B, grad et liste theta
    n = len(theta0)
    B = np.eye(n)

    # Première étape
    grad =  Function1.grad_oracle(L,theta0)
    p = -eta*grad
    theta1 = theta0 + p
    liste_theta = [theta0, theta1]

    while (np.linalg.norm(liste_theta[-1] - liste_theta[-2])) > eps :
        # On calcule les nouvelles valeurs
        grad_new = Function1.grad_oracle(L,liste_theta[-1])
        theta_new = liste_theta[-1] - eta*np.dot(B,grad)
        s = theta_new - liste_theta[-1]
        y = grad_new - grad
        Bs = np.dot(B,s)
        B = B + np.dot(y,y.T)/np.dot(y,s) - np.dot(Bs,Bs.T)/np.dot(s,Bs)
        
        # On met à jour les anciennes valeurs
        liste_theta.append(theta_new)
        grad = grad_new
        
    print("Nombre d'étapes : "+ str(len(liste_theta)))
    return liste_theta[-1]


def BfgsDescent1(eps, theta0,L,eta):
    # On definit les variables : n, B, grad et liste theta
    n = len(theta0)
    B = np.eye(n)

    # Première étape
    grad =  Function1.grad_oracle(L,theta0)
    p = -eta*grad
    theta1 = theta0 + p
    liste_theta = [theta0, theta1]

    while (np.linalg.norm(liste_theta[-1] - liste_theta[-2])) > eps :
        # On calcule les nouvelles valeurs
        grad_new = Function1.grad_oracle(L,liste_theta[-1])
        theta_new = liste_theta[-1] - eta*np.dot(B,grad)
        s = theta_new - liste_theta[-1]
        y = grad_new - grad
        Bs = np.dot(B,s)
        B = B + np.dot(y,y.T)/np.dot(y,s) - np.dot(Bs,Bs.T)/np.dot(s,Bs)
        
        # On met à jour les anciennes valeurs
        liste_theta.append(theta_new)
        grad = grad_new
        
    return len(liste_theta)

"""
def StochasticDescent(eps, theta0, L, eta):
    # On fait une première étape hors de la boucle
    X = Function2.value_oracle(L,theta0)
    randomized_grad = 
    p = -eta*Function1.grad_oracle(L, theta0)
    theta1 = theta0 + p
    liste_theta = [theta0, theta1]
    while (np.linalg.norm(liste_theta[-1] - liste_theta[-2])) > eps :
"""
         


