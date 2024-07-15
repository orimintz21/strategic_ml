class StrategicHingeLoss(nn.Module):
    def __init__(self, lambda_reg: float = 1.0) -> None:
        super(StrategicHingeLoss, self).__init__()
        self.lambda_reg = lambda_reg

    def forward(self, x: Tensor, y: Tensor, y_tilde: Tensor, w: Tensor) -> Tensor:
        # Calculate the hinge loss with strategic adjustments
        hinge_loss: Tensor = torch.clamp(
            1 - y * (torch.matmul(x, w) + 2 * y_tilde * torch.norm(w, p=2)), min=0
        )
        # Regularization term
        reg_term: Tensor = self.lambda_reg * torch.norm(w, p=2) ** 2
        return hinge_loss.mean() + reg_term
