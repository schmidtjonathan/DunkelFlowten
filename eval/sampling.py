import torch
import tqdm.auto as tqdm


def edm_noise_schedule(nfes: int, rho=7):
    step_indices = torch.arange(nfes, dtype=torch.float64)
    sigma_min = 0.002
    sigma_max = 80.0
    sigma_vec = (
        sigma_max ** (1 / rho)
        + step_indices / (nfes - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    sigma_vec = torch.cat([sigma_vec, torch.zeros_like(sigma_vec[:1])])
    return sigma_vec


def edm_noise2time(noise):
    return 1.0 - torch.clip(noise / (1 + noise), min=0.0, max=1.0)


def edm_time_discretization(nfes: int, rho=7):
    sigma_vec = edm_noise_schedule(nfes, rho)
    time_vec = edm_noise2time(sigma_vec.squeeze())
    return time_vec


def sampler(
    net,
    x0,
    extra={},
    num_steps=18,
    rho=7,
    midpoint=False,
):
    # Time step discretization.
    time_steps = edm_time_discretization(num_steps, rho=rho).to(torch.float64)
    # jacs = []
    # Main sampling loop.
    x_next = x0
    assert len(time_steps) == num_steps + 1
    with torch.no_grad():
        for i, (t_cur, t_next) in (
            bar := tqdm.tqdm(
                enumerate(zip(time_steps[:-1], time_steps[1:]), 1), total=num_steps
            )
        ):
            torch.cuda.empty_cache()
            x_cur = x_next

            dt = t_next - t_cur
            assert dt > 0
            bar.set_description(f"t = {t_cur:.5f} +++ dt={dt:.5f}, NFE={net.get_nfe()}")

            # u_prior_1, u_lik_1 = net(x=x_cur, t=t_cur)
            # jacs.append(u_lik_1)
            # u_1 = u_prior_1 + u_lik_1
            u_1 = net(x=x_cur, t=t_cur,
                    #   extra=extra
                      )

            if midpoint and (i < num_steps - 1):
                half_step = dt / 2
                # u_prior_2, u_lik_2 = net(x=x_cur + half_step * u_1, t=t_cur + half_step)
                # u_2 = u_prior_2 + u_lik_2
                u_2 = net(x=x_cur + half_step * u_1, t=t_cur + half_step,
                        #   extra=extra
                          )
                x_next = x_cur + dt * u_2
            else:
                x_next = x_cur + dt * u_1

            if torch.any(torch.isnan(x_next)):
                raise RuntimeError("NaN in sample")

    return x_next  # , torch.stack(jacs).cpu()
