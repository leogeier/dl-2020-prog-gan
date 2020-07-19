import torch


class Loss:

    def __init__(self, discriminator):
        self.discriminator = discriminator

    def discriminator_loss(self, real_samples, fake_samples, current_depth, alpha):
        raise NotImplementedError

    def generator_loss(self, real_samples, fake_samples, current_depth, alpha):
        raise NotImplementedError


class WassersteinLoss(Loss):

    def __init__(self, discriminator, drift=0.001, use_gradient_penalty=False):
        super(WassersteinLoss, self).__init__(discriminator)
        self.drift = drift
        self.use_gradient_penalty = use_gradient_penalty

    def __gradient_penalty(self, real_samples, fake_samples, current_depth, alpha, reg_lambda=10):
        batch_size = real_samples.shape[0]

        epsilon = torch.rand((batch_size, 1, 1, 1)).to(fake_samples.device)

        merged = epsilon * real_samples + ((1 - epsilon) * fake_samples)
        merged.requires_grad_(True)

        output = self.discriminator(merged, current_depth, alpha)

        gradient = torch.autograd.grad(outputs=output,
                                       inputs=merged,
                                       grad_outputs=torch.ones_like(output),
                                       create_graph=True,
                                       retain_graph=True,
                                       only_inputs=True)[0]
        gradient = gradient.view(gradient.shape[0], -1)

        return reg_lambda * torch.mean(gradient.norm(p=2, dim=1).sub(1).pow(2))

    def discriminator_loss(self, real_samples, fake_samples, current_depth, alpha):
        output_for_fakes = self.discriminator(fake_samples, current_depth, alpha)
        output_for_real = self.discriminator(real_samples, current_depth, alpha)

        loss = output_for_fakes.mean() - output_for_real.mean() + self.drift * output_for_real.pow(2).mean()

        if self.use_gradient_penalty:
            penalty = self.__gradient_penalty(real_samples, fake_samples, current_depth, alpha)
            loss += penalty

        return loss

    def generator_loss(self, real_samples, fake_samples, current_depth, alpha):
        return - torch.mean(self.discriminator(fake_samples, current_depth, alpha))
