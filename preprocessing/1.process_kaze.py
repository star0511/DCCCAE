import argparse
import functools
from typing import Optional
from dataset import *
from util import process_dataset_images
from preprocessing.normalize import preprocess_signature

def process_dataset(dataset: IterableDataset,
                    save_path: str,
                    subset: Optional[slice] = None):
    """ Processes a dataset (normalizing the images) and saves the result as a
        numpy npz file (collection of np.ndarrays).

    Parameters
    ----------
    dataset : IterableDataset
        The dataset, that knows where the signature files are located
    save_path : str
        The name of the file to save the numpy arrays
    img_size : tuple (H x W)
        The final size of the images
    subset : slice
        Which users to consider. e.g. slice(None) to consider all users, or slice(first, last)


    Returns
    -------
    None

    """
    preprocess_fn = functools.partial(preprocess_signature,
                                      canvas_size=dataset.maxsize,
                                      img_size=dataset.maxsize,
                                      input_size=dataset.maxsize)  # Don't crop it now
    if subset is None:
        subset = slice(None)  # Use all
    processed = process_dataset_images(dataset, preprocess_fn, dataset.maxsize, subset)
    x, y, yforg, user_mapping, used_files = processed

    np.savez(save_path,
             x=x,
             y=y,
             yforg=yforg,
             user_mapping=user_mapping,
             filenames=used_files)


if __name__ == '__main__':
    available_datasets = {
        'CEDAR': CedarDataset,
        'UTSig': UTSigDataset,
        'BHSigB': BHSigBDataset,
        'BHSigH': BHSigHDataset,
        'SigComp2011': SigCompDataset
    }
    for datasetname in available_datasets.keys():
        ds = available_datasets[datasetname]
        dataset = ds()

        print('Processing dataset')
        process_dataset(dataset, "../data/"+datasetname+"maxsize")
