import torch


class WassersteinLoss:
    """
    Implements the improved Wasserstein loss with gradient penalty (WGAN-GP) as suggested by Gulrajani et al.
    (https://arxiv.org/pdf/1704.00028.pdf) and as used in the NVIDIA progressive gan paper. The original objective
    function is described in equation 2 and the regularizing gradient penalty term in equation 3.
    The Wasserstein loss encourages the critic / discriminator to separate output for real and fake images
    respectively, i.e. large values for real images, small values for fakes. The generator is encouraged to produce
    fake images with high scores (realistic ones).
    This penalty is needed to keep the gradient lengths in a compact interval (Lipschitz constraint). This in turn is
    a mathematical property required by the Wasserstein loss.
    Additionally this class uses the penalty term with a small weight (eps_drift) for large critic / discriminator
    output suggested in the progressive gan paper to keep output reasonably sized.
    """

    def __init__(self, discriminator, eps_drift=0.001, use_gp=False):
        """
        :param discriminator: discriminator module of the GAN being trained
        :param eps_drift: weight for output size penalty
        """
        self.discriminator = discriminator
        self.eps_drift = eps_drift
        self.use_gp = use_gp

    def __gradient_penalty(self, real_samples, fake_samples, current_depth, alpha, gp_lambda=10):
        batch_size = real_samples.shape[0]

        epsilon = torch.rand((batch_size, 1, 1, 1)).to(fake_samples.device)

        # interpolate between real and fake images by a random factor epsilon
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

        return gp_lambda * torch.mean(gradient.norm(p=2, dim=1).sub(1).pow(2))

    def discriminator_loss(self, real_samples, fake_samples, current_depth, alpha):
        output_for_fakes = self.discriminator(fake_samples, current_depth, alpha)
        output_for_real = self.discriminator(real_samples, current_depth, alpha)

        # original Wasserstein loss (fake output counts positive to encourage minimization and vice versa)
        loss = output_for_fakes.mean() - output_for_real.mean()
        if self.use_gp:
            gradient_penalty = self.__gradient_penalty(real_samples, fake_samples, current_depth, alpha)
            loss += gradient_penalty  # improved Wasserstein loss with gradient penalty
        loss += self.eps_drift * output_for_real.pow(2).mean()  # drift constraint from progressive GAN

        return loss

    def generator_loss(self, fake_samples, current_depth, alpha):
        return - torch.mean(self.discriminator(fake_samples, current_depth, alpha))


class ConditionalWLoss:

    def __init__(self, discriminator, eps_drift=0.001, use_gp=False):
        """
        :param discriminator: discriminator module of the GAN being trained
        :param eps_drift: weight for output size penalty
        """
        assert discriminator.conditional, "Can't use conditional loss with unconditional discriminator"
        self.discriminator = discriminator
        self.eps_drift = eps_drift
        self.use_gp = use_gp

    def __gradient_penalty(self, attributes, real_samples, fake_samples, current_depth, alpha, gp_lambda=10):
        batch_size = real_samples.shape[0]

        epsilon = torch.rand((batch_size, 1, 1, 1)).to(fake_samples.device)

        # interpolate between real and fake images by a random factor epsilon
        merged = epsilon * real_samples + ((1 - epsilon) * fake_samples)
        merged.requires_grad_(True)

        output = self.discriminator(merged, current_depth, alpha, attributes)

        gradient = torch.autograd.grad(outputs=output,
                                       inputs=merged,
                                       grad_outputs=torch.ones_like(output),
                                       create_graph=True,
                                       retain_graph=True,
                                       only_inputs=True)[0]
        gradient = gradient.view(gradient.shape[0], -1)

        return gp_lambda * torch.mean(gradient.norm(p=2, dim=1).sub(1).pow(2))

    def discriminator_loss(self, attributes, real_samples, fake_samples, current_depth, alpha):
        output_for_fakes = self.discriminator(fake_samples, current_depth, alpha, attributes)
        output_for_real = self.discriminator(real_samples, current_depth, alpha, attributes)

        # original Wasserstein loss (fake output counts positive to encourage minimization and vice versa)
        loss = output_for_fakes.mean() - output_for_real.mean()
        if self.use_gp:
            gradient_penalty = self.__gradient_penalty(attributes, real_samples, fake_samples, current_depth, alpha)
            loss += gradient_penalty  # improved Wasserstein loss with gradient penalty
        loss += self.eps_drift * output_for_real.pow(2).mean()  # drift constraint from progressive GAN

        return loss

    def generator_loss(self, attributes, fake_samples, current_depth, alpha):
        return - torch.mean(self.discriminator(fake_samples, current_depth, alpha, attributes))
