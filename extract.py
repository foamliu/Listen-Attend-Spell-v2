import tarfile


def extract(filename):
    print('Extracting {}...'.format(filename))
    tar = tarfile.open(filename, 'r')
    tar.extractall('data')
    tar.close()


if __name__ == "__main__":
    # if not os.path.isdir('data/faces_ms1m_112x112'):
    extract('data/data_thchs30.tgz')
