# list_classes.py
from data_loader import class_names


print(f"Total classes: {len(class_names)}\n")
for idx, name in enumerate(class_names):
    print(f"{idx+1:3d}. {name}")


