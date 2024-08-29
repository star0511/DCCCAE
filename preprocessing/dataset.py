from abc import ABC, abstractmethod
from typing import List, Iterable, Tuple
import numpy as np
import os
from skimage.io import imread
from skimage import img_as_ubyte
class IterableDataset(ABC):
    """ Base class for Datasets. The dataset classes specify information about
     the dataset (such as the number of signatures per user), as well as implement
     methods to iterate over the signatures of an user. """

    @property
    def maxsize(self):
        """ Maximum size of the images in the dataset"""
        raise NotImplementedError

    @property
    def genuine_per_user(self):
        """ Number of genuine signatures per user"""
        raise NotImplementedError

    @property
    def skilled_per_user(self):
        """ Number of skilled forgeries per user"""
        raise NotImplementedError

    @property
    def simple_per_user(self):
        """ Number of simple forgeries per user"""
        raise NotImplementedError

    @abstractmethod
    def get_user_list(self) -> List[int]:
        """ Returns the list of users in the dataset. """
        pass

    @abstractmethod
    def iter_genuine(self, user: int) -> Iterable[Tuple[np.ndarray, str]]:
        """ Iterates through the genuine signatures of an user

        Parameters
        ----------
        user : int
            The user, from whom the signatures will be returned

        Returns
        -------
        Generator (iterable) of <image, filename>

        """
        pass

    @abstractmethod
    def iter_simple_forgery(self, user: int) -> Iterable[Tuple[np.ndarray, str]]:
        """ Iterates through the simple forgeries of an user

        Parameters
        ----------
        user : int
            The user, from whom the signatures will be returned

        Returns
        -------
        Generator (iterable) of <image, filename>

        """
        pass

    @abstractmethod
    def iter_forgery(self, user: int) -> Iterable[Tuple[np.ndarray, str]]:
        """ Iterates through the skilled forgeries of an user

        Parameters
        ----------
        user : int
            The user, from whom the signatures will be returned

        Returns
        -------
        Generator (iterable) of <image, filename>

        """
        pass
class CedarDataset(IterableDataset):
    """ Helper class to load the CEDAR dataset
    """
    def __init__(self):
        self.path = "../dataset/CEDAR"
        self.users = list(range(1, 55+1))

    @property
    def genuine_per_user(self):
        return 24

    @property
    def skilled_per_user(self):
        return 24

    @property
    def simple_per_user(self):
        return 0

    @property
    def maxsize(self):
        return 952, 1360
        return 730, 1042

    def get_user_list(self):
        return self.users

    def iter_genuine(self, user):
        """ Iterate over genuine signatures for the given user"""

        files = ['{}_{}_{}.png'.format('original', user, img) for img in range(1, 24 + 1)]
        for f in files:
            full_path = os.path.join(self.path, 'full_org', f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def iter_forgery(self, user):
        """ Iterate over skilled forgeries for the given user"""

        files = ['{}_{}_{}.png'.format('forgeries', user, img) for img in range(1, 24 + 1)]
        for f in files:
            full_path = os.path.join(self.path, 'full_forg', f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def iter_simple_forgery(self, user):
        yield from ()  # No simple forgeries
class UTSigDataset(IterableDataset):
    """ Helper class to load the UTSig dataset
    """
    def __init__(self):
        self.path = "../dataset/UTSig"
        self.users = list(range(1, 115+1))

    @property
    def genuine_per_user(self):
        return 27

    @property
    def skilled_per_user(self):
        return 6

    @property
    def simple_per_user(self):
        return 0

    @property
    def maxsize(self):
        return 1500, 2100

    def get_user_list(self):
        return self.users

    def iter_genuine(self, user):
        """ Iterate over genuine signatures for the given user"""

        files = ['{}/{}.tif'.format(user, img) for img in range(1, 27 + 1)]
        for f in files:
            full_path = os.path.join(self.path,"Genuine", f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def iter_forgery(self, user):
        """ Iterate over skilled forgeries for the given user"""


        files = ['{}/{}.tif'.format(user, img) for img in range(1, 6 + 1)]
        for f in files:
            full_path = os.path.join(self.path,"Forgery","Skilled", f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def iter_simple_forgery(self, user):
        yield from ()  # No simple forgeries
class BHSigBDataset(IterableDataset):
    """ Helper class to load the UTSig dataset
    """
    def __init__(self):
        self.path = "../dataset/BHSig260"
        self.users = list(range(1, 100+1))

    @property
    def genuine_per_user(self):
        return 24

    @property
    def skilled_per_user(self):
        return 30

    @property
    def simple_per_user(self):
        return 0

    @property
    def maxsize(self):
        return 310, 1100

    def get_user_list(self):
        return self.users

    def iter_genuine(self, user):
        """ Iterate over genuine signatures for the given user"""

        files = ['{}{}{}/B-S-{}-G-{}{}.tif'.format(int(user/100%10),int(user/10%10),int(user%10),user, int(img/10%10),int(img%10)) for img in range(1, 24 + 1)]
        for f in files:
            full_path = os.path.join(self.path,"Bengali", f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def iter_forgery(self, user):
        """ Iterate over skilled forgeries for the given user"""

        files = ['{}{}{}/B-S-{}-F-{}{}.tif'.format(int(user / 100 % 10), int(user / 10 % 10), int(user % 10), user,
                                                   int(img / 10 % 10), int(img % 10)) for img in range(1, 30 + 1)]
        for f in files:
            full_path = os.path.join(self.path, "Bengali", f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def iter_simple_forgery(self, user):
        yield from ()  # No simple forgeries
class BHSigHDataset(IterableDataset):
    """ Helper class to load the UTSig dataset
    """
    def __init__(self):
        self.path = "../dataset/BHSig260"
        self.users = list(range(1, 160+1))

    @property
    def genuine_per_user(self):
        return 24

    @property
    def skilled_per_user(self):
        return 30

    @property
    def simple_per_user(self):
        return 0

    @property
    def maxsize(self):
        return 450, 1350

    def get_user_list(self):
        return self.users

    def iter_genuine(self, user):
        """ Iterate over genuine signatures for the given user"""

        files = ['{}{}{}/H-S-{}-G-{}{}.tif'.format(int(user/100%10),int(user/10%10),int(user%10),user, int(img/10%10),int(img%10)) for img in range(1, 24 + 1)]
        for f in files:
            full_path = os.path.join(self.path,"Hindi", f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def iter_forgery(self, user):
        """ Iterate over skilled forgeries for the given user"""

        files = ['{}{}{}/H-S-{}-F-{}{}.tif'.format(int(user / 100 % 10), int(user / 10 % 10), int(user % 10), user,
                                                   int(img / 10 % 10), int(img % 10)) for img in range(1, 30 + 1)]
        for f in files:
            full_path = os.path.join(self.path, "Hindi", f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def iter_simple_forgery(self, user):
        yield from ()  # No simple forgeries
class SigCompDataset(IterableDataset):
    """ Helper class to load the SigComp dataset
     """

    def __init__(self):
        self.path = "/home/ubuntu/CCBR/dataset/SigComp2011"
        self.users = list(range(1, 20 + 1))

    @property
    def genuine_per_user(self):
        return 21

    @property
    def skilled_per_user(self):
        return 23

    @property
    def simple_per_user(self):
        return 0

    @property
    def maxsize(self):
        return 800, 1300

    def get_user_list(self):
        return self.users

    def iter_genuine(self, user):
        """ Iterate over genuine signatures for the given user"""

        files = ['{}_{}.png'.format(user, img) for img in range(1, 21 + 1)]
        for f in files:
            full_path = os.path.join(self.path, 'G', f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def iter_forgery(self, user):
        """ Iterate over skilled forgeries for the given user"""

        files = ['{}_{}.png'.format(user, img) for img in range(1, 23 + 1)]
        for f in files:
            full_path = os.path.join(self.path, 'F', f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def iter_simple_forgery(self, user):
        yield from ()  # No simple forgeries