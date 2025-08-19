import os
import os.path
from io import BytesIO

import tqdm
from PIL import Image
from torchvision.datasets.vision import VisionDataset


class LvisDetection(VisionDataset):
    """`LVIS Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root,
        annFile,
        transform=None,
        target_transform=None,
        transforms=None,
        cache_mode=False,
        local_rank=0,
        local_size=1,
    ):
        super(LvisDetection, self).__init__(root, transforms, transform, target_transform)
        from lvis import LVIS

        self.lvis = LVIS(annFile)
        self.ids = list(sorted(self.lvis.imgs.keys()))
        self.category_ids = self.lvis.cats.keys()
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size
        if cache_mode:
            self.cache = {}
            self.cache_images()
            # map coco discrete category ids to contiguous ids
        self.category_to_class = {
            c: i + 1
            for i, c in enumerate(sorted(self.category_ids))
        }
        self.class_to_category = {
            i + 1: c
            for i, c in enumerate(sorted(self.category_ids))
        }

        for img in self.lvis.imgs.values():
            img['aspect_ratio'] = float(img['height']) / img['width']
        self.aspect_ratios = [self.lvis.imgs[ix]['aspect_ratio'] for ix in self.ids]

    def cache_images(self):
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            path = self.lvis.load_imgs(img_id)[0]["file_name"]
            with open(os.path.join(self.root, path), "rb") as f:
                self.cache[path] = f.read()

    def get_image(self, path):
        if self.cache_mode:
            if path not in self.cache.keys():
                with open(os.path.join(self.root, path), "rb") as f:
                    self.cache[path] = f.read()
            return Image.open(BytesIO(self.cache[path])).convert("RGB")
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    @property
    def num_images_per_category(self):
        if hasattr(self, '_num_images_per_category'):
            return self._num_images_per_category
        self._num_images_per_category = {
            cat: len(set(img_list))
            for cat, img_list in self.lvis.cat_img_map.items()
        }
        return self._num_images_per_category

    @property
    def num_images_per_class(self):
        """ For Class Aware Balanced Sampler and Repeat Factor Sampler
        """
        return {
            self.category_to_class[cat]: num
            for cat, num in self.num_images_per_category.items()
        }

    def get_image_classes(self, img_index):
        """For Repeat Factor Sampler
        """
        img_id = self.ids[img_index]
        img_anns = self.lvis.img_ann_map[img_id]
        return [
            self.category_to_class[ann['category_id']]
            for ann in img_anns if not ann.get('iscrowd', False)
        ]

    def __getitem__(self, index):
        lvis = self.lvis
        img_id = self.ids[index]
        ann_ids = lvis.get_ann_ids(img_ids=[img_id])
        target = lvis.load_anns(ann_ids)

        split_folder, file_name = lvis.load_imgs([img_id])[0]["coco_url"].split("/")[-2:]
        path = os.path.join(split_folder, file_name)

        img = self.get_image(path)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)
