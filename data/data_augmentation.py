from albumentations import (
    Compose, ImageCompression, HueSaturationValue, RandomBrightnessContrast, HorizontalFlip,
    Rotate, CoarseDropout, VerticalFlip, PixelDropout
)

pixel_transforms = Compose([
    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.5),
    CoarseDropout(max_holes=10, max_height=20, max_width=20,
                  min_holes=1, min_height=5, min_width=5),
    PixelDropout(0.1),
    ImageCompression(quality_lower=50, quality_upper=100, p=0.5),
])

spatial_transforms = Compose([
    # RandomResizedCrop(512,512, scale=(0.6,1.0)),
    Rotate(limit=90),
    HorizontalFlip(),
    VerticalFlip(),
], additional_targets={"depth": "image"})


def aug_fn(image, depth):
    data = {"image": image, "depth": depth}
    aug_data = spatial_transforms(**data)
    aug_data = pixel_transforms(**aug_data)
    return aug_data["image"], aug_data["depth"]
