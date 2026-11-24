from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

folder = Path(r"D:\rjh07\archive (1)")

# 图片文件前16字节是说明信息，跳过后读取第一张图的784个像素
pixels = np.fromfile(
    folder / "train-images.idx3-ubyte",
    dtype=np.uint8,
    offset=16,
    count=784,
)

# 标签文件前8字节是说明信息，跳过后读取第一个标签
labels = np.fromfile(
    folder / "train-labels.idx1-ubyte",
    dtype=np.uint8,
    offset=8,
    count=1,
)

image = pixels.reshape(28, 28)
label = int(labels[0])

print("图片形状：", image.shape)
print("正确标签：", label)

plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()