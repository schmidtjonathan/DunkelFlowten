import torch
import numpy as np
from sda_atmos.flow.utils import ModelWrapper
from sda_atmos.flow.path.affine import CondOTProbPath
import tqdm.auto as tqdm
import gc


def unfold_cond(x, k):
    w = 2 * k + 1
    return x.unfold(0, w, 1).movedim(-1, 1)


def unfold(x, k):
    w = 2 * k + 1
    x = x.unfold(0, w, 1)
    x = x.movedim(-1, 1)
    x = x.flatten(1, 2)
    return x


def fold(x, k, mode="full"):
    w = 2 * k + 1
    x = x.unflatten(1, (w, -1))

    assert mode in ["full", "first", "middle", "last"]
    if mode == "full":
        return torch.cat([x[0, :k], x[:, k], x[-1, -k:]], dim=0)
    elif mode == "first":
        return torch.cat([x[0, :k], x[:, k]], dim=0)
    elif mode == "middle":
        return x[:, k]
    elif mode == "last":
        return torch.cat([x[:, k], x[-1, -k:]], dim=0)
    else:
        raise ValueError(f"Unknown fold mode {mode}")


def get_device(device):
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    return device


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BatchedFlow(torch.autograd.Function):
    @staticmethod
    def setup_context(ctx, args, output):
        flow, x, t, markov_order, batch_size, month_cond, hour_cond = args
        ctx.flow = flow
        ctx.markov_order = markov_order
        ctx.batch_size = batch_size
        ctx.month_cond = month_cond
        ctx.hour_cond = hour_cond
        ctx.save_for_backward(x, t)

    @staticmethod
    def forward(flow, x, t, markov_order, batch_size, month_cond, hour_cond):
        with torch.no_grad():
            if batch_size is None:
                return fold(
                    flow(
                        x=unfold(x, markov_order),
                        t=t,
                        month_cond=unfold_cond(month_cond, markov_order),
                        hour_cond=unfold_cond(hour_cond, markov_order),
                    ),
                    markov_order,
                )

            # print("batch size", batch_size)
            assert batch_size > 0
            x_batches = unfold(x, markov_order).split(batch_size, dim=0)
            num_batches = len(x_batches)

            model_device = next(flow.parameters()).device
            t = t.to(model_device)

            month_cond_batches = unfold_cond(
                month_cond.to(model_device), markov_order
            ).split(batch_size, dim=0)
            hour_cond_batches = unfold_cond(
                hour_cond.to(model_device), markov_order
            ).split(batch_size, dim=0)
            assert len(month_cond_batches) == len(
                hour_cond_batches) == num_batches

            out_windows = []
            for batch_index in tqdm.tqdm(range(num_batches), leave=False, disable=True):
                foldmode = (
                    "full"
                    if num_batches == 1
                    else (
                        "first"
                        if batch_index == 0
                        else ("last" if batch_index == num_batches - 1 else "middle")
                    )
                )
                # print(f"Batch {batch_index+1}/{num_batches} -> {foldmode} ->", end=" ")
                current_window = x_batches[batch_index].to(model_device)
                current_month_cond = month_cond_batches[batch_index]
                current_hour_cond = hour_cond_batches[batch_index]
                current_window = fold(
                    flow(
                        x=current_window,
                        t=t,
                        month_cond=current_month_cond,
                        hour_cond=current_hour_cond,
                    ).cpu(),
                    markov_order,
                    mode=foldmode,
                )
                # print(f"{current_window.shape}")
                out_windows.append(current_window)

            # out_full = torch.cat(out_windows, 0)
            # print("out-full shape", out_full.shape)
            # return fold(out_full, markov_order)
            return torch.cat(out_windows, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        # print("BACKWARD")
        x, t = ctx.saved_tensors
        flow = ctx.flow
        markov_order = ctx.markov_order
        batch_size = ctx.batch_size
        month_cond = ctx.month_cond
        hour_cond = ctx.hour_cond
        w = 2 * markov_order + 1

        if batch_size is None:
            # For full batch processing

            with torch.enable_grad():
                x = x.clone().detach().requires_grad_(True)
                output = fold(
                    flow(
                        x=unfold(x, markov_order),
                        t=t,
                        month_cond=unfold_cond(month_cond, markov_order),
                        hour_cond=unfold_cond(hour_cond, markov_order),
                    ),
                    markov_order,
                )
                output.backward(grad_output)
                grad_x = x.grad
        else:
            # Initialize gradients
            grad_x = torch.zeros_like(x, device=torch.device("cpu"))

            # Create windows similar to forward pass
            x_batches = unfold(x, markov_order).split(batch_size, 0)
            num_batches = len(x_batches)
            # print(f"[b] {num_batches} batches: {x_batches[0].shape}, ..., {x_batches[-1].shape}")

            # Split grad_output according to the forward pass output structure
            grad_splits = []
            current_idx = 0

            for batch_idx in range(num_batches):
                is_first = batch_idx == 0
                is_last = batch_idx == num_batches - 1
                batch_size_current = x_batches[batch_idx].size(0)

                if is_first:
                    split_size = markov_order + batch_size_current
                elif is_last:
                    split_size = batch_size_current + markov_order
                else:
                    split_size = batch_size_current

                grad_splits.append(
                    grad_output[current_idx: current_idx + split_size])
                current_idx += split_size

            # ---
            model_device = next(flow.parameters()).device
            t = t.to(model_device)

            month_cond_batches = unfold_cond(
                month_cond.to(model_device), markov_order
            ).split(batch_size, dim=0)
            hour_cond_batches = unfold_cond(
                hour_cond.to(model_device), markov_order
            ).split(batch_size, dim=0)
            assert len(month_cond_batches) == len(
                hour_cond_batches) == num_batches

            # Process each batch
            for batch_idx in range(num_batches):
                is_first = batch_idx == 0
                is_last = batch_idx == num_batches - 1
                # print(f"Batch {batch_idx+1}/{num_batches} ->", end=" ")

                # Move batch to GPU
                current_x = x_batches[batch_idx].to(device)
                current_grad = grad_splits[batch_idx].to(device)
                current_month_cond = month_cond_batches[batch_idx]
                current_hour_cond = hour_cond_batches[batch_idx]

                # Reshape gradient to match the window structure
                if is_first or is_last:
                    grad_window = torch.zeros(
                        current_x.size(0),
                        w,
                        current_x.size(1) // w,
                        current_x.size(2),
                        current_x.size(3),
                        device=device,
                    )

                    if is_first:
                        grad_window[0,
                                    :markov_order] = current_grad[:markov_order]
                        grad_window[:,
                                    markov_order] = current_grad[markov_order:]
                    elif is_last:
                        grad_window[:,
                                    markov_order] = current_grad[:-markov_order]
                        grad_window[-1, -
                                    markov_order:] = current_grad[-markov_order:]
                else:
                    grad_window = torch.zeros_like(
                        current_x.unflatten(1, (w, -1)))
                    grad_window[:, markov_order] = current_grad

                # Reshape back to match unet input format
                grad_window = grad_window.flatten(1, 2)

                # Compute gradients for this batch
                with torch.enable_grad():
                    current_x.requires_grad_(True)

                    # Forward pass for this batch
                    output = flow(
                        x=current_x,
                        t=t,
                        month_cond=current_month_cond,
                        hour_cond=current_hour_cond,
                    )

                    # Backward pass
                    output.backward(grad_window)

                # Accumulate gradients
                start_idx = batch_idx * batch_size

                if start_idx < grad_x.size(0):
                    # Unfold the gradient to match the original window structure
                    grad_current = current_x.grad.unflatten(
                        1, (w, -1)
                    )  # [B, w, C, H, W]

                    # Accumulate gradients for each position in the window
                    for window_pos in range(w):
                        target_idx = start_idx + window_pos
                        if target_idx < grad_x.size(0):
                            valid_batch_indices = torch.arange(
                                min(current_x.size(0),
                                    grad_x.size(0) - target_idx)
                            )
                            grad_x[
                                target_idx: target_idx +
                                len(valid_batch_indices)
                            ] += grad_current[valid_batch_indices, window_pos].to(
                                torch.device("cpu")
                            )

                # Clear GPU memory
                del current_x, current_grad, grad_window
                torch.cuda.empty_cache()

        return None, grad_x, None, None, None, None, None


class FlowModel(ModelWrapper):
    def __init__(self, model):
        super().__init__(model)
        self.model = model
        self.nfe_counter = 0

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        month_cond: torch.Tensor,
        hour_cond: torch.Tensor,
    ) -> torch.Tensor:
        t = torch.zeros(x.shape[0], device=x.device) + t

        result = self.model(x, t, dict(hour=hour_cond, month=month_cond))

        self.nfe_counter += 1
        return result

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter


class PosteriorSequenceModel(ModelWrapper):
    def __init__(
        self,
        model,
        markov_order,
        A,
        y,
        month_cond,
        hour_cond,
        std=1e-2,
        gamma=1e-2,
        clip_target=False,
        exact_grad=False,
        batch_size=None,
        device=None,
    ):
        super().__init__(model)
        self.A = A
        self.y = y

        self.markov_order = markov_order
        self.window_size = 2 * markov_order + 1

        self.std = std
        self.gamma = gamma
        self.path = CondOTProbPath()
        self.exact_grad = exact_grad

        self.batch_size = batch_size
        self.clip_target = clip_target

        self.batch_size = batch_size
        self.device = get_device(device)

        self.month_cond = month_cond
        self.hour_cond = hour_cond

    def reset_nfe_counter(self) -> None:
        self.model.reset_nfe_counter()

    def get_nfe(self) -> int:
        return self.model.get_nfe()

    @property
    def is_batched(self):
        return self.batch_size is not None

    def forward(self, x, t):
        sched_stats = self.path.scheduler(t)
        alpha = sched_stats.alpha_t
        sigma = sched_stats.sigma_t
        d_alpha = sched_stats.d_alpha_t
        d_sigma = sched_stats.d_sigma_t

        if not self.exact_grad:
            prior = BatchedFlow.apply(
                self.model,
                x,
                t,
                self.markov_order,
                self.batch_size,
                self.month_cond,
                self.hour_cond,
            )

        with torch.enable_grad():
            x = x.clone().detach().requires_grad_(True)

            if self.exact_grad:
                prior = BatchedFlow.apply(
                    self.model,
                    x,
                    t,
                    self.markov_order,
                    self.batch_size,
                    self.month_cond,
                    self.hour_cond,
                )

            x_ = self.path.velocity_to_target(prior, x, t)
            res = self.y - self.A(x_)
            var = self.std**2 + self.gamma * (sigma / alpha) ** 2

            logp = -(res**2 / var).sum() / 2

        (s,) = torch.autograd.grad(logp, x)
        prior_eps = self.path.velocity_to_epsilon(prior, x, t)
        lik_eps = -sigma * s

        # return prior, -((sigma * d_sigma * alpha - (sigma**2) * d_alpha) / alpha) * s
        return self.path.epsilon_to_velocity(prior_eps + lik_eps, x, t)


def test_batched():
    rnd_x = torch.randn(26, 4, 32, 32, dtype=torch.float32)
    rnd_t = torch.rand((1,), dtype=torch.float32)
    months = torch.LongTensor(np.random.choice(11, (26,)))
    hours = torch.LongTensor(np.random.choice(3, (26,)))

    flow_model = FlowModel(model).eval()
    print(next(flow_model.parameters()).dtype)

    order = args_dict["markov_order"]

    with torch.no_grad():
        out_expected = (
            fold(
                flow_model(
                    unfold(rnd_x.clone().to("cuda"), order),
                    rnd_t.clone().to("cuda"),
                    month_cond=unfold_cond(months, order).to("cuda"),
                    hour_cond=unfold_cond(hours, order).to("cuda"),
                ),
                order,
                mode="full",
            )
            .detach()
            .cpu()
        )
        out_full = (
            BatchedFlow.apply(
                flow_model,
                rnd_x.clone().to("cuda"),
                rnd_t.clone().to("cuda"),
                order,
                None,
                months.to("cuda"),
                hours.to("cuda"),
            )
            .detach()
            .cpu()
        )

        print(out_expected.dtype, out_full.dtype)
        torch.testing.assert_close(out_full, out_expected)
        print("Full batch works")

        out_batch = (
            BatchedFlow.apply(
                flow_model,
                rnd_x.clone(),
                rnd_t.clone(),
                order,
                6,
                months,
                hours,
            )
            .detach()
            .cpu()
        )

        torch.testing.assert_close(out_full, out_batch)
        print("Mini batch works")

    del out_full, out_batch
    torch.cuda.empty_cache()
    gc.collect()

    # --

    def DX(x, t, batch_size):
        return BatchedFlow.apply(
            flow_model,
            x,
            t,
            order,
            batch_size,
            months.to("cuda"),
            hours.to("cuda"),
        ).mean()

    def DX2(x, t):
        return fold(
            flow_model(
                unfold(x, order),
                t,
                unfold_cond(months, order).to("cuda"),
                unfold_cond(hours, order).to("cuda"),
            ),
            order,
        ).mean()

    # grad_expected = torch.func.jacrev(DX, argnums=0, chunk_size=1)(
    #     rnd_x.clone().to("cuda"),
    #     rnd_t.clone().to("cuda"),
    #     None,
    # ).detach().cpu()
    grad_batch = (
        torch.func.jacrev(DX, argnums=0, chunk_size=1)(
            rnd_x.clone(), rnd_t.clone(), 6)
        .detach()
        .cpu()
    )
    grad_expected = (
        torch.func.jacrev(DX2, argnums=0, chunk_size=1)(
            rnd_x.clone().to("cuda"),
            rnd_t.clone().to("cuda"),
        )
        .detach()
        .cpu()
    )

    torch.testing.assert_close(grad_expected, grad_batch, rtol=1e-3, atol=1e-5)

    del grad_batch, grad_expected
    torch.cuda.empty_cache()
    gc.collect()
