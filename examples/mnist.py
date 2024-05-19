import equinox as eqx
import jax
import jax.numpy as jnp
import optax  # https://github.com/deepmind/optax
import torch  # https://pytorch.org
import torchvision  # https://pytorch.org
import torchvision.transforms as T
import plotly.graph_objs as go
import numpy as np
import time
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping
from kanx import KAN
from typing import List


# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 3e-3
STEPS = 100
PRINT_EVERY = 50
SEED = 5678


class KAN_classifier(eqx.Module):
    kan: KAN

    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 5,
        use_base_update: bool = True,
        base_activation=jax.nn.silu,
        spline_weight_init_scale: float = 0.1,
        *,
        key,
    ) -> None:
        super().__init__()
        self.kan = KAN(
            layers_hidden=layers_hidden,
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids,
            use_base_update=use_base_update,
            base_activation=base_activation,
            spline_weight_init_scale=spline_weight_init_scale,
            key=key,
        )

    def __call__(self, x):
        x = self.kan(x)
        x = jax.nn.log_softmax(x)
        return x


normalise_data = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
        T.Lambda(lambda x: torch.flatten(x)),
    ]
)
train_dataset = torchvision.datasets.MNIST(
    "MNIST",
    train=True,
    download=True,
    transform=normalise_data,
)
test_dataset = torchvision.datasets.MNIST(
    "MNIST",
    train=False,
    download=True,
    transform=normalise_data,
)
trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True
)

dummy_x, dummy_y = next(iter(trainloader))
dummy_x = dummy_x.numpy()
dummy_y = dummy_y.numpy()

key = jax.random.PRNGKey(0)
in_features = 28 * 28
out_features = 10
model = KAN_classifier(layers_hidden=[in_features, 64, out_features], key=key)


def loss(
    model, x: Float[Array, "batch 28*28"], y: Int[Array, " batch"]
) -> Float[Array, ""]:
    # Our input has the shape (BATCH_SIZE, 28*28), but our model operations on
    # a single input input image of shape (28*28).
    #
    # Therefore, we have to use jax.vmap, which in this case maps our model over the
    # leading (batch) axis.
    pred_y = jax.vmap(model)(x)
    return cross_entropy(y, pred_y)


def cross_entropy(
    y: Int[Array, " batch"], pred_y: Float[Array, "batch 10"]
) -> Float[Array, ""]:
    # y are the true targets, and should be integers 0-9.
    # pred_y are the log-softmax'd predictions.
    pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
    return -jnp.mean(pred_y)


optim = optax.adamw(LEARNING_RATE)
loss = eqx.filter_jit(loss)  # JIT our loss function from earlier!


@eqx.filter_jit
def compute_accuracy(
    model: KAN, x: Float[Array, "batch 1 28 28"], y: Int[Array, " batch"]
) -> Float[Array, ""]:
    """This function takes as input the current model
    and computes the average accuracy on a batch.
    """
    pred_y = jax.vmap(model)(x)
    pred_y = jnp.argmax(pred_y, axis=1)
    return jnp.mean(y == pred_y)


def evaluate(model: KAN, testloader: torch.utils.data.DataLoader):
    """This function evaluates the model on the test dataset,
    computing both the average loss and the average accuracy.
    """
    avg_loss = 0
    avg_acc = 0
    for x, y in testloader:
        x = x.numpy()
        y = y.numpy()
        # Note that all the JAX operations happen inside `loss` and `compute_accuracy`,
        # and both have JIT wrappers, so this is fast.
        avg_loss += loss(model, x, y)
        avg_acc += compute_accuracy(model, x, y)
    return avg_loss / len(testloader), avg_acc / len(testloader)


def train(
    model: KAN,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    optim: optax.GradientTransformation,
    steps: int,
    print_every: int,
) -> KAN:
    # Just like earlier: It only makes sense to train the arrays in our model,
    # so filter out everything else.
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    def make_step(
        model: KAN,
        opt_state: PyTree,
        x: Float[Array, "batch 1 28*28"],
        y: Int[Array, " batch"],
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    # Loop over our training dataset as many times as we need.
    def infinite_trainloader():
        while True:
            yield from trainloader
    test_accuracy_report = []
    test_loss_report = []
    for step, (x, y) in zip(range(steps), infinite_trainloader()):
        # PyTorch dataloaders give PyTorch tensors by default,
        # so convert them to NumPy arrays.
        x = x.numpy()
        y = y.numpy()
        model, opt_state, train_loss = make_step(model, opt_state, x, y)
        if (step % print_every) == 0 or (step == steps - 1):
            test_loss, test_accuracy = evaluate(model, testloader)
            test_loss_report.append(test_loss)
            test_accuracy_report.append(test_accuracy)
            print(
                f"{step=}, train_loss={train_loss.item()}, "
                f"test_loss={test_loss.item()}, test_accuracy={test_accuracy.item()}"
            )

    return model, test_accuracy_report, test_loss_report


key = jax.random.PRNGKey(SEED)

s = time.time()
model, test_accuracy_report, test_loss_report = train(model, trainloader, testloader, optim, STEPS, PRINT_EVERY)
print(f"Time to train: {time.time()-s}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(test_accuracy_report)), y=test_accuracy_report, mode='lines', name='lines'))
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(test_loss_report)), y=test_loss_report, mode='lines', name='lines'))
fig.show()