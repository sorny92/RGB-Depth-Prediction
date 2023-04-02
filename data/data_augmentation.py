from albumentations import (
    Compose, RandomBrightness, ImageCompression, HueSaturationValue, RandomBrightnessContrast, HorizontalFlip,
    Rotate, CoarseDropout, VerticalFlip, PixelDropout, RandomResizedCrop
)

pixel_transforms = Compose([
    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    CoarseDropout(),
    PixelDropout(0.1),
    ImageCompression(quality_lower=30, quality_upper=100, p=0.5),
])

spatial_transforms = Compose([
    RandomResizedCrop(512,512, scale=(0.6,1.0)),
    Rotate(limit=180),
    HorizontalFlip(),
    VerticalFlip(),
], additional_targets={"depth": "image"})


def aug_fn(image, depth):
    data = {"image": image, "depth": depth}
    aug_data = spatial_transforms(**data)
    aug_data = pixel_transforms(**aug_data)
    return aug_data["image"], aug_data["depth"]
