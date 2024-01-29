class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'arcade':
            return './data/arcade/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
