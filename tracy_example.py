from tracy import Profiler


def test_fn():
    import ttnn
    import torch

    x = torch.rand(1, 2, 3, 4)
    x_tt = ttnn.as_tensor(x)
    return


if __name__ == "__main__":
    profiler = Profiler()
    profiler.enable()
    test_fn()
    profiler.disable()
