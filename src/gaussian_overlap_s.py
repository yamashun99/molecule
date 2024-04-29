import numpy as np
from scipy.special import erf


class GaussianOverlapFunction:

    @staticmethod
    def F(t):
        return 0.5 * np.sqrt(np.pi / (t + 1e-10)) * erf(np.sqrt(t + 1e-10))

    @staticmethod
    def T(alpha, beta, RA, RB):
        alpha = np.atleast_1d(alpha)
        beta = np.atleast_1d(beta)
        alpha2 = alpha[:, None]
        beta2 = beta[None, :]
        return np.squeeze(
            alpha2
            * beta2
            / (alpha2 + beta2)
            * (
                6
                - 4
                * alpha2
                * beta2
                / (alpha2 + beta2)
                * np.linalg.norm(RA - RB, axis=-1) ** 2
            )
            * (np.pi / (alpha2 + beta2)) ** (3 / 2)
            * np.exp(
                -alpha2
                * beta2
                * np.linalg.norm(RA - RB, axis=-1) ** 2
                / (alpha2 + beta2)
            )
            / 2
        )

    @staticmethod
    def V(alpha, beta, RA, RB, RC):
        alpha = np.atleast_1d(alpha)
        beta = np.atleast_1d(beta)
        RP = (
            alpha[:, None, None] * RA[None, None, :]
            + beta[None, :, None] * RB[None, None, :]
        ) / (alpha[:, None, None] + beta[None, :, None])
        RP_RC_norm_sq = np.linalg.norm(RP - RC[None, None, :], axis=2) ** 2
        gamma = alpha[:, None] + beta[None, :]
        return np.squeeze(
            -2
            * np.pi
            / gamma
            * np.exp(
                -alpha[:, None] * beta[None, :] * np.linalg.norm(RA - RB) ** 2 / gamma
            )
            * GaussianOverlapFunction.F(gamma * RP_RC_norm_sq)
        )

    @staticmethod
    def U(alpha, beta, gamma, delta, RA, RB, RC, RD):
        alpha = np.atleast_1d(alpha)
        beta = np.atleast_1d(beta)
        gamma = np.atleast_1d(gamma)
        delta = np.atleast_1d(delta)

        alpha5 = alpha[:, None, None, None, None]
        beta5 = beta[None, :, None, None, None]
        gamma5 = gamma[None, None, :, None, None]
        delta5 = delta[None, None, None, :, None]
        RA5 = RA[None, None, None, None, :]
        RB5 = RB[None, None, None, None, :]
        RC5 = RC[None, None, None, None, :]
        RD5 = RD[None, None, None, None, :]
        RP = (alpha5 * RA5 + gamma5 * RC5) / (alpha5 + gamma5)
        RQ = (beta5 * RB5 + delta5 * RD5) / (beta5 + delta5)
        RP_RQ_norm_sq = np.linalg.norm(RP - RQ, axis=-1) ** 2
        alpha4 = alpha[:, None, None, None]
        beta4 = beta[None, :, None, None]
        gamma4 = gamma[None, None, :, None]
        delta4 = delta[None, None, None, :]

        return np.squeeze(
            2
            * np.pi ** (5 / 2)
            / (
                (alpha4 + gamma4)
                * (beta4 + delta4)
                * np.sqrt(alpha4 + gamma4 + beta4 + delta4)
            )
            * np.exp(
                -alpha4 * gamma4 * np.linalg.norm(RA - RC) ** 2 / (alpha4 + gamma4)
                - beta4 * delta4 * np.linalg.norm(RB - RD) ** 2 / (beta4 + delta4)
            )
            * GaussianOverlapFunction.F(
                (alpha4 + gamma4)
                * (beta4 + delta4)
                * RP_RQ_norm_sq
                / (alpha4 + gamma4 + beta4 + delta4)
            )
        )

    # @staticmethod
    # def U(alpha, beta, gamma, delta, RA, RB, RC, RD):
    #    RP = (alpha * RA + gamma * RC) / (alpha + gamma)
    #    RQ = (beta * RB + delta * RD) / (beta + delta)
    #    return (
    #        2
    #        * np.pi ** (5 / 2)
    #        / ((alpha + gamma) * (beta + delta) * np.sqrt(alpha + gamma + beta + delta))
    #        * np.exp(
    #            -alpha * gamma * np.linalg.norm(RA - RC) ** 2 / (alpha + gamma)
    #            - beta * delta * np.linalg.norm(RB - RD) ** 2 / (beta + delta)
    #        )
    #        * GaussianOverlapFunction.F(
    #            (alpha + gamma)
    #            * (beta + delta)
    #            * np.linalg.norm(RP - RQ) ** 2
    #            / (alpha + gamma + beta + delta)
    #        )
    #    )

    @staticmethod
    def S(alpha, beta, RA, RB):
        return (np.pi / (alpha + beta)) ** (3 / 2) * np.exp(
            -alpha * beta / (alpha + beta) * np.linalg.norm(RA - RB) ** 2
        )
