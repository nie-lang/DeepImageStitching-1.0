import os
import argparse
import configparser


def get_dir(directory):
    """
    get the directory, if no such directory, then make it.

    @param directory: The new directory.
    """

    if not os.path.exists(directory):
        os.makedirs(directory)

    return directory


def parser_args():
    parser = argparse.ArgumentParser(description='Options to run the network.')
    parser.add_argument('-g', '--gpu', type=str, default='0',
                        help='the device id of gpu.')
    parser.add_argument('-i', '--iters', type=int, default=1,
                        help='set the number of iterations, default is 1')
    parser.add_argument('-b', '--batch', type=int, default=1,
                        help='set the batch size, default is 4.')
    parser.add_argument('--train_folder', type=str, default='',
                        help='set the training folder path.')
    parser.add_argument('--test_folder', type=str, default='',
                        help='set the testing folder path.')
    parser.add_argument('--snapshot_dir', type=str, default='',
                        help='if it is folder, then it is the directory to save models, '
                             'if it is a specific model.ckpt-xxx, then the system will load it for testing.')
    parser.add_argument('--summary_dir', type=str, default='', help='the directory to save summaries.')

    return parser.parse_args()


class Const(object):
    class ConstError(TypeError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("Can't change const.{}".format(name))
        if not name.isupper():
            raise self.ConstCaseError('const name {} is not all uppercase'.format(name))

        self.__dict__[name] = value

    def __str__(self):
        _str = '<================ Constants information ================>\n'
        for name, value in self.__dict__.items():
            print(name, value)
            _str += '\t{}\t{}\n'.format(name, value)

        return _str


args = parser_args()
const = Const()

# inputs constants
const.TRAIN_FOLDER = args.train_folder
const.TEST_FOLDER = args.test_folder

const.GPU = args.gpu

const.BATCH_SIZE = args.batch
const.ITERATIONS = args.iters



const.SNAPSHOT_DIR = get_dir(args.snapshot_dir) 
const.SUMMARY_DIR = get_dir(args.summary_dir)



