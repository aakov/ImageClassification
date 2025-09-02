import re
from collections import defaultdict
from data_loader import class_names, train_images

def normalize_class_name(name):

    # remove year (4 digits at the end)
    collapsed = re.sub(r'\s\d{4}$', '', name)
    return collapsed

def build_class_mapping():

    new_name_to_id = {}
    old_to_new = {}
    new_class_names = []

    for old_id, name in enumerate(class_names):
        collapsed_name = normalize_class_name(name)
        if collapsed_name not in new_name_to_id:
            new_id = len(new_name_to_id)
            new_name_to_id[collapsed_name] = new_id
            new_class_names.append(collapsed_name)
        old_to_new[old_id] = new_name_to_id[collapsed_name]

    return old_to_new, new_class_names

def remap_images(train_images, old_to_new):

    for img in train_images:
        img.label = old_to_new[img.label - 1]  # -1 since labels are 1-based in the dataset
    return train_images

if __name__ == "__main__":
    print(f"Original classes: {len(class_names)}")
    old_to_new, new_class_names = build_class_mapping()
    print(f"Collapsed classes: {len(new_class_names)}")

    # Count images per new class
    counts = defaultdict(int)
    for img in train_images:
        new_label = old_to_new[img.label - 1]
        counts[new_label] += 1

    # Show distribution
    for i, name in enumerate(new_class_names):
        print(f"{i}. {name} -> {counts[i]} images")
