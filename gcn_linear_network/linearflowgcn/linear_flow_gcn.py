"""
    First attempt at runing the linear gcn network on pytorch with pytorch lightning.
"""
import logging
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import time

from torch_geometric import transforms
from torch_geometric.nn import GCNConv
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric.data import Data
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
import numpy as np
import wandb

# these change if the datagen changes.
logging.basicConfig(level=logging.INFO)
NODE_FEATURES = 2
NODE_EMBEDDINGS_1 = 128
NODE_EMBEDDINGS_2 = 64
# EDGE_ATTRIBUTES = 2 # TODO dont use
MLP_INPUT = NODE_EMBEDDINGS_2 * 2  # stack two nodes
MLP_EMBEDDINGS_1 = NODE_EMBEDDINGS_2  # half the input
MLP_EMBEDDINGS_2 = int(NODE_EMBEDDINGS_2/2)  # half the input
OUTPUT_SIZE = 1
num_arcs = 2
num_nodes = 3
gcn_size = 64
mlp_size = 64

def calculate_mean_std(dataloader:DataLoader):
    with torch.no_grad():
        alldata = torch.cat(tuple(b.x for b in dataloader))
        meanvec = alldata.mean(0)
        stdvec = alldata.std(0)
        return meanvec,stdvec
    # with torch.no_grad():
    #     node_feature_accs = torch.tensor(NODE_FEATURES)
    #     num_nodes = 0
    #     for b in dataloader:
    #         node_feature_accs = node_feature_accs+b.x.sum(0)
    #         num_nodes += b.x.shape[0]
    #     mean = node_feature_accs/num_nodes
    #
    #     stdevs = 0.0
    #     for b in dataloader:
    #         stdevs += ((b.x - mean)**2).sum(0)
    #     stdevs = stdevs/num_nodes
    #     stdevs = torch.sqrt(stdevs)
    #     return mean,stdevs


def generate_flow_data(num_examples, min_demand, max_demand):
    """
    Generates the training data of a three node MCF linear program. sample runif the demand and split runif
    on the two inventory nodes, to make the network learn the structure of the graph.
    Should be a trivial example at best.

    :param num_examples: number of training examples to generate
    :param min_demand:
    :param max_demand:
    :return:
    """
    datalist = []
    for i in range(num_examples):
        total_demand = float(np.random.randint(min_demand, max_demand)) * 1.0
        first_path_flow_pct = np.random.uniform(0.0, 1.0)
        second_path_flow_pct = 1 - first_path_flow_pct

        first_path_inventory = int(total_demand * first_path_flow_pct) * 1.0
        second_path_inventory = total_demand - first_path_inventory

        assert (first_path_inventory + second_path_inventory) == total_demand

        # total_demand = 20
        if total_demand % 2 == 1:
            total_demand += 1
        nodes = torch.tensor(
            [
                [first_path_inventory, 1.0],
                [second_path_inventory, 1.0],
                [-total_demand, 2.0],
            ],
            dtype=torch.float,
        )

        edge_index = torch.tensor([[0, 2], [1, 2],], dtype=torch.long)

        edge_attr = torch.tensor(
            [[first_path_inventory], [second_path_inventory],],  # todo not sure if used
            dtype=torch.float,
        )

        y_edges = torch.tensor([[first_path_flow_pct], [second_path_flow_pct]])

        d1 = Data(x=nodes, edge_index=edge_index, edge_attr=edge_attr, y_edges=y_edges)
        datalist.append(d1)
    return datalist


def MLP(channels, batch_norm=True):
    return Seq(
        *[
            Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
            for i in range(1, len(channels))
        ]
    )


class LinearFlowGCN(pl.LightningModule):
    """
        A network that attempts to learn the solution of a linear program by applying two GCN layers
        and two MLP layers to learn the normalized flow of two arcs that should sum to one.
    """

    def __init__(self, solved_epsilon, learning_rate, batch_size,meanvec,stdvec, verbose=False):
        # NODE_FEATURES = 2
        # NODE_EMBEDDINGS_1 = 32
        # NODE_EMBEDDINGS_2 = 16
        # #EDGE_ATTRIBUTES = 2
        # MLP_INPUT = NODE_EMBEDDINGS_2*2
        # MLP_EMBEDDINGS_1 = NODE_EMBEDDINGS_2 # half the input
        super(LinearFlowGCN, self).__init__()
        self.transform = transforms.NormalizeFeatures()
        self.conv1 = GCNConv(NODE_FEATURES, NODE_EMBEDDINGS_1)
        self.bn1 = torch.nn.BatchNorm1d(NODE_EMBEDDINGS_1)
        self.conv2 = GCNConv(NODE_EMBEDDINGS_1, NODE_EMBEDDINGS_2)
        self.bn2 = torch.nn.BatchNorm1d(NODE_EMBEDDINGS_2)
        self.lin1 = MLP([MLP_INPUT, MLP_EMBEDDINGS_1])
        self.lin2 = MLP([MLP_EMBEDDINGS_1, MLP_EMBEDDINGS_2])
        self.lin3 = MLP([MLP_EMBEDDINGS_2, OUTPUT_SIZE])

        self.verbose = verbose
        self.solved_epsilon = solved_epsilon
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.meanvec=meanvec
        self.stdvec=stdvec

    def forward(self, data):
        # data = self.transform(data)
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch_size = int(x.shape[0] / num_nodes)

        # normalizing
        edge_attr = edge_attr / edge_attr.sum()
        x = (x - self.meanvec) / self.stdvec

        #### GCN ####
        x = self.conv1(
            x, edge_index
        )  # receives the edge indices to determine neighbors
        x = self.bn1(x)  # receives the edge indices to determine neighbors
        x = F.relu(x)

        # x = F.dropout(x,training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)  # receives the edge indices to determine neighbors
        x = F.relu(x)

        #### MLP ####
        # pass MLP to each pair of node embeddings
        arc_embeddings = []
        for g in range(batch_size):
            # HARDCODED ARC STACKING, todo see if there's a better way to do this.

            node_index_0 = (g * num_nodes) + 0
            node_index_1 = (g * num_nodes) + 1
            node_index_2 = (g * num_nodes) + 2
            arc_index_0 = (g * num_arcs) + 0
            arc_index_1 = (g * num_arcs) + 1
            # fmt: off
            arc_embedding = torch.stack(
                # First arc
                [torch.cat((x[node_index_0], x[node_index_1])),

                # Second arc
                torch.cat((x[node_index_1], x[node_index_2]))]
            )
            # fmt: on
            arc_embeddings.append(arc_embedding)
        arc_stack = torch.cat(
            arc_embeddings
        )  # (batch_size,nodefeatures*4+edge_features*2)
        fcn_arc_stack = self.lin1(arc_stack)
        fcn_arc_stack = F.relu(fcn_arc_stack)

        fcn_arc_stack = self.lin2(fcn_arc_stack)  # (batch_size,num_arcs)
        fcn_arc_stack = F.relu(fcn_arc_stack)

        fcn_arc_stack = self.lin3(fcn_arc_stack)
        out = torch.tanh(fcn_arc_stack)

        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def calculate_solved(self, y_hat, y):
        return (F.l1_loss(y_hat, y) < self.solved_epsilon).int()

    def training_step(self, batch, batch_idx):
        y = batch.y_edges
        y_hat = self(batch).clamp(0, 1)
        loss = F.binary_cross_entropy(y_hat, y)
        solved = self.calculate_solved(y_hat, y)
        result = pl.TrainResult(loss)
        result.log("train_loss", loss)
        result.log("train_solved", solved)
        return result

    def validation_step(self, batch, batch_idx):
        y = batch.y_edges
        y_hat = self(batch)
        loss = F.binary_cross_entropy(y_hat.clamp(0, 1), y)
        solved = self.calculate_solved(y_hat, y)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log("val_loss", loss)
        result.log("val_solved", solved)
        return result

    def test_step(self, batch, batch_idx):
        logging.info("Testing")
        y = batch.y_edges
        y_hat = self(batch)
        loss = F.binary_cross_entropy(y_hat.clamp(0, 1), y)
        if self.verbose:
            y_hat_d = y_hat.detach().flatten().reshape(int(y.shape[0]/2),2)
            y_d = y.detach().flatten().reshape(int(y.shape[0]/2),2)

            for i in range(y_hat_d.shape[0]):
                mse = F.mse_loss(y_hat_d[i], y_d[i])
                mae = F.l1_loss(y_hat_d[i], y_d[i])
                logging.info(
                    f"Test: {y_hat_d[i].tolist()} expected {y_d[i].tolist()} (Loss: {loss} MSE: {mse}) MAE: {mae}"
                )
        result = pl.EvalResult(checkpoint_on=loss)
        solved = self.calculate_solved(y_hat, y)
        result.log("test_loss", loss)
        result.log("test_solved", solved)
        return result


if __name__ == "__main__":
    config_dict = {
        "training_examples": 20000,
        # "training_examples": 100,
        "test_examples": 64,
        "learning_rate": 1e-5,
        # the best that have worked so far. TODO:  Do a sweep when the new architecture is ready
        "batch_size": 256,  # TODO SWEEP
        "max_epochs": 200,
        "min_demand": 10,
        "max_demand": 100,
        "watch_gradients": True,
        "solved_epsilon": 1e-5,  # difference in l1 loss to consider the LP solved.
        "gradient_clip_val": 0.9
    }
    wandb.init(config=config_dict)
    config = wandb.config  # turn into an object

    experiment_name = f"lfgcnv4_{config.max_epochs}epochs_gradientclip_512bs_2"

    # Startup wandb logger.
    wandb_logger = WandbLogger(
        name=experiment_name,
        project="linearflowgcn",
        tags=[
            "experiment",
             # "debug"
        ],
    )
    logging.info(f"Running script for experiment {experiment_name}")
    print("Good old print")

    # Create data and model
    training_data = generate_flow_data(
        num_examples=config.training_examples,
        min_demand=config.min_demand,
        max_demand=config.max_demand,
    )
    test_data = generate_flow_data(
        num_examples=config.test_examples,
        min_demand=config.min_demand,
        max_demand=config.max_demand,
    )

    val_data = generate_flow_data(
        num_examples=config.test_examples,
        min_demand=config.min_demand,
        max_demand=config.max_demand,
    )
    data_loader = DataLoader(training_data, batch_size=config.batch_size)
    val_loader = DataLoader(val_data, batch_size=config.batch_size)
    test_loader = DataLoader(test_data, batch_size=config.batch_size)
    meanvec,stdvec = calculate_mean_std(training_data)
    model = LinearFlowGCN(
        config.solved_epsilon, config.learning_rate, config.batch_size,meanvec,stdvec, verbose=True
    )

    if config.watch_gradients:
        logging.info("Watching gradients")
        wandb_logger.watch(model, log="gradients", log_freq=100)
    wandb_logger.log_hyperparams(dict(config))

    # Setup PL Trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        logger=wandb_logger,
        default_root_dir=os.path.join(os.getcwd(), "models/"),
        log_save_interval=1,
        gradient_clip_val=config.gradient_clip_val
    )

    # Fitting and testing
    logging.info(f"Fitting model with config: {config}")
    trainer.fit(model, data_loader, val_loader)

    logging.info("Calling test on fitted model")
    trainer.test(test_dataloaders=test_loader)

    logging.info("Done")
