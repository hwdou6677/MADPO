"""TRPO utility functions."""
import torch


def flat_grad(grads):
    """Flatten the gradients."""
    grad_flatten = []
    for grad in grads:
        if grad is None:
            continue
        grad_flatten.append(grad.view(-1))
    grad_flatten = torch.cat(grad_flatten)
    return grad_flatten


def GaussianMatrix(x, y, sigma):
    size_x = x.size()
    size_y = y.size()
    G = (x * x).sum(-1)
    H = (y * y).sum(-1)
    Q = G.unsqueeze(-1).repeat(1, size_y[0])
    R = H.unsqueeze(-1).T.repeat(size_x[0], 1)
    H = Q + R - 2 * x @ y.T
    H = torch.exp(-H / 2 / sigma ** 2)
    return H


def CCSD(obs_current, obs_pre, action_current, action_pre, device="cuda:0", sigma=1):
    obs_current = torch.tensor(obs_current).float().to(device)
    obs_pre = torch.tensor(obs_pre).float().to(device)
    # action_current = torch.tensor(action_current)
    action_pre = torch.tensor(action_pre).float().to(device)

    K1 = GaussianMatrix(obs_current, obs_current, sigma)
    K2 = GaussianMatrix(obs_pre, obs_pre, sigma)

    L1 = GaussianMatrix(action_current, action_current, sigma)
    L2 = GaussianMatrix(action_pre, action_pre, sigma)

    K12 = GaussianMatrix(obs_current, obs_pre, sigma)
    L12 = GaussianMatrix(action_current, action_pre, sigma)

    K21 = GaussianMatrix(obs_pre, obs_current, sigma)
    L21 = GaussianMatrix(action_pre, action_current, sigma)

    H1 = K1 * L1
    self_term1 = (H1.sum(-1) / ((K1.sum(- 1)) ** 2)).sum(0)

    H2 = K2 * L2
    self_term2 = (H2.sum(-1) / ((K2.sum(- 1)) ** 2)).sum(0)

    H3 = K12 * L12
    cross_term1 = (H3.sum(-1) / ((K1.sum(-1)) * (K12.sum(-1)))).sum(0)

    H4 = K21 * L21
    cross_term2 = (H4.sum(-1) / ((K2.sum(-1)) * (K21.sum(-1)))).sum(0)

    cs1 = -2 * torch.log2(cross_term1) + torch.log2(self_term1) + torch.log2(self_term2)
    cs2 = -2 * torch.log2(cross_term2) + torch.log2(self_term1) + torch.log2(self_term2)

    cs = ((cs1 + cs2) / 2)

    return cs


def flat_hessian(hessians):
    """Flatten the hessians."""
    hessians_flatten = []
    for hessian in hessians:
        if hessian is None:
            continue
        hessians_flatten.append(hessian.contiguous().view(-1))
    hessians_flatten = torch.cat(hessians_flatten).data
    return hessians_flatten


def flat_params(model):
    """Flatten the parameters."""
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    params_flatten = torch.cat(params)
    return params_flatten


def update_model(model, new_params):
    """Update the model parameters."""
    index = 0
    for params in model.parameters():
        params_length = len(params.view(-1))
        new_param = new_params[index: index + params_length]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += params_length


def kl_approx(p, q):
    """KL divergence between two distributions."""
    r = torch.exp(q - p)
    kl = r - 1 - q + p
    return kl


def _kl_normal_normal(p, q):
    """KL divergence between two normal distributions.
    adapted from https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html#kl_divergence
    """
    var_ratio = (p.scale.to(torch.float64) / q.scale.to(torch.float64)).pow(2)
    # print(var_ratio)
    t1 = (
            (p.loc.to(torch.float64) - q.loc.to(torch.float64)) / q.scale.to(torch.float64)
    ).pow(2)
    x = 0.5 * (var_ratio + t1 - 1 - var_ratio.log())
    # print(x)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())


def kl_divergence(
        obs,
        rnn_states,
        action,
        masks,
        available_actions,
        active_masks,
        new_actor,
        old_actor,
):
    """KL divergence between two distributions."""
    _, _, new_dist = new_actor.evaluate_actions(
        obs, rnn_states, action, masks, available_actions, active_masks
    )
    with torch.no_grad():
        _, _, old_dist = old_actor.evaluate_actions(
            obs, rnn_states, action, masks, available_actions, active_masks
        )
    if new_dist.__class__.__name__ == "FixedCategorical":  # discrete action
        new_logits = new_dist.logits
        old_logits = old_dist.logits
        kl = kl_approx(old_logits, new_logits)
    else:  # continuous action
        kl = _kl_normal_normal(old_dist, new_dist)

    if len(kl.shape) > 1:
        kl = kl.sum(1, keepdim=True)
    return kl


def kl_grad(new_dist, old_dist):
    if new_dist.__class__.__name__ == "FixedCategorical":  # discrete action
        new_logits = new_dist.logits
        old_logits = old_dist.logits
        kl = kl_approx(old_logits, new_logits)
    else:  # continuous action
        kl = _kl_normal_normal(old_dist, new_dist)

    if len(kl.shape) > 1:
        kl = kl.sum(1, keepdim=True)
    return kl

# pylint: disable-next=invalid-name
def conjugate_gradient(
        actor,
        obs,
        rnn_states,
        action,
        masks,
        available_actions,
        active_masks,
        b,
        nsteps,
        device,
        residual_tol=1e-10,
):
    """Conjugate gradient algorithm.
    # refer to https://github.com/openai/baselines/blob/master/baselines/common/cg.py
    """
    x = torch.zeros(b.size()).to(device=device)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(nsteps):
        _Avp = fisher_vector_product(
            actor, obs, rnn_states, action, masks, available_actions, active_masks, p
        )
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x


def fisher_vector_product(
        actor, obs, rnn_states, action, masks, available_actions, active_masks, p
):
    """Fisher vector product."""
    with torch.backends.cudnn.flags(enabled=False):
        p.detach()
        kl = kl_divergence(
            obs,
            rnn_states,
            action,
            masks,
            available_actions,
            active_masks,
            new_actor=actor,
            old_actor=actor,
        )
        kl = kl.mean()
        kl_grad = torch.autograd.grad(
            kl, actor.parameters(), create_graph=True, allow_unused=True
        )
        kl_grad = flat_grad(kl_grad)  # check kl_grad == 0
        kl_grad_p = (kl_grad * p).sum()
        kl_hessian_p = torch.autograd.grad(
            kl_grad_p, actor.parameters(), allow_unused=True
        )
        kl_hessian_p = flat_hessian(kl_hessian_p)
        return kl_hessian_p + 0.1 * p
