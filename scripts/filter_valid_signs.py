import os

batch_dirs = [
    'data/batch_2',
    'data/batch_13',
    'data/batch_23',
    'data/batch_33',
    'data/batch_43'
]

for batch in batch_dirs:
    if os.path.exists(batch):
        print(f"\n🔎 {batch}:")
        files = os.listdir(batch)
        for f in files[:5]:
            print(f"  ➤ {f}")
    else:
        print(f"❌ Folder not found: {batch}")
