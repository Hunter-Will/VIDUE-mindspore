from importlib import import_module
import mindspore.dataset as ds


class Data:
    def __init__(self, args):
        self.args = args
        self.data_train = args.data_train
        self.data_test = args.data_test

        # load training dataset
        if not self.args.test_only:
            m_train = import_module('data.' + self.data_train.lower())
            trainset = getattr(m_train, self.data_train.upper())(self.args, name=self.data_train, train=True)
            self.loader_train = ds.GeneratorDataset(
                trainset,
                column_names=["inputs","gts","out_names","in_names","exposure"],
                shuffle=True,
                num_parallel_workers=self.args.n_threads,
            ).batch(self.args.batch_size,drop_remainder=True).create_tuple_iterator(do_copy=False)
        else:
            self.loader_train = None

        # load testing dataset
        m_test = import_module('data.' + self.data_test.lower())
        testset = getattr(m_test, self.data_test.upper())(self.args, name=self.data_test, train=False)
        self.loader_test = ds.GeneratorDataset(
            testset,
            column_names=["inputs","gts","out_names","in_names","exposure"],
            shuffle=False,
            num_parallel_workers=self.args.n_threads
        ).batch(1).create_tuple_iterator(do_copy=False)
