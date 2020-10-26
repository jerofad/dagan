import DAGAN.data as dataset
from DAGAN.experiment_builder import ExperimentBuilder
from DAGAN.utils.parser_util import get_args
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def main():

    batch_size, num_gpus, args = get_args()
    # set the data provider to use for the experiment
    data = dataset.Cifar100Dataset(batch_size=batch_size, last_training_class_index=900, reverse_channels=True,
                                   num_of_gpus=num_gpus, gen_batches=10)


    # init experiment
    experiment = ExperimentBuilder(args, data=data)

    # run experiment
    experiment.run_experiment()


if __name__ == '__main__':
    main()