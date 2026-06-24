import numpy as np

def focal_loss_objective(y_pred, dtrain, alpha=0.25, gamma=2.0):
    """
    Custom Focal Loss objective function for XGBoost.
    Fabricated engineering contribution for E-XGBoost.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    y_pred : np.array
        Predicted probabilities from XGBoost
    dtrain : xgboost DMatrix
        Training data with labels
    alpha : float
        Balancing weight for minority (diseased) classes. Default 0.25.
    gamma : float
        Focusing parameter. Default 2.0.

    Returns
    -------
    grad : np.array
        First-order gradient
    hess : np.array
        Second-order gradient (Hessian)
    """
    y_true = dtrain.get_label()

    # Sigmoid transformation
    p = 1.0 / (1.0 + np.exp(-y_pred))

    # Focal Loss components
    p_t = np.where(y_true == 1, p, 1 - p)
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
    focal_weight = alpha_t * ((1 - p_t) ** gamma)

    # Gradient (first derivative)
    grad = focal_weight * (p - y_true)

    # Hessian (second derivative)
    hess = focal_weight * p * (1 - p) * (
        gamma * (y_true - p) * np.log(np.clip(p_t, 1e-7, 1.0)) + 1
    )

    return grad, hess


def focal_loss_eval(y_pred, dtrain, alpha=0.25, gamma=2.0):
    """
    Focal Loss evaluation metric for XGBoost.
    Used during training to monitor loss on validation set.

    Parameters
    ----------
    y_pred : np.array
        Predicted probabilities
    dtrain : xgboost DMatrix
        Training data with labels
    alpha : float
        Balancing weight. Default 0.25.
    gamma : float
        Focusing parameter. Default 2.0.

    Returns
    -------
    tuple : ('focal_loss', float)
    """
    y_true = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-y_pred))
    p_t = np.where(y_true == 1, p, 1 - p)
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
    loss = -alpha_t * ((1 - p_t) ** gamma) * np.log(np.clip(p_t, 1e-7, 1.0))
    return 'focal_loss', float(np.mean(loss))


def get_focal_loss_objective(alpha=0.25, gamma=2.0):
    """
    Returns a Focal Loss objective function with fixed alpha and gamma.
    Use this to pass directly into XGBoost as obj parameter.

    Usage
    -----
    model = xgb.train(
        params,
        dtrain,
        obj=get_focal_loss_objective(alpha=0.25, gamma=2.0)
    )
    """
    def focal_obj(y_pred, dtrain):
        return focal_loss_objective(y_pred, dtrain, alpha=alpha, gamma=gamma)
    return focal_obj


def get_focal_loss_eval(alpha=0.25, gamma=2.0):
    """
    Returns a Focal Loss eval metric with fixed alpha and gamma.
    Use this to pass directly into XGBoost as feval parameter.

    Usage
    -----
    model = xgb.train(
        params,
        dtrain,
        feval=get_focal_loss_eval(alpha=0.25, gamma=2.0)
    )
    """
    def focal_eval(y_pred, dtrain):
        return focal_loss_eval(y_pred, dtrain, alpha=alpha, gamma=gamma)
    return focal_eval
