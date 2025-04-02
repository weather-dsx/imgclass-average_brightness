### 功能
- 加载预训练的`resnet-18`对图片进行分类
- 用PIL和OpenCV读取图片三个通道的平均亮度
- 可视化

---

### 使用

- 待分类图片放入`input_img`文件夹
- 运行`SE1_1.py`
- 图片分类结果及平均亮度计算结果输出在`classify_img`

---
### 文件夹结构
```cmd
C:.
│  imagenet_classes.txt \\ imagenet类别
│  SE1.py  \\第一版代码
│  SE1_1.py \\第二版代码
│  小作业1.md \\报告文档
│  小作业1.pdf \\ 报告pdf版
│
├─classify_img \\输出图片文件夹
│      1739949031750_ski mask_classified.jpg
│      20210101142830_flagpole_classified.jpg
│      CRnall_20250223_145939162_snow leopard_classified.jpg
│      CRnall_20250223_160123879_ambulance_classified.jpg
│      OIP-C_goldfish_classified.jpg
│      R-C_cabbage butterfly_classified.jpg
│      Screenshot_2025-02-26-19-01-11-413_Persian cat_classified.jpg
│      Top10_palace_classified.jpg
│
├─input_img \\ 输入图片文件夹
│      1739949031750.jpg
│      20210101142830.jpg
│      CRnall_20250223_145939162.jpg
│      CRnall_20250223_160123879.jpg
│      OIP-C.jpg
│      R-C.jpg
│      Screenshot_2025-02-26-19-01-11-413.png
│      Top10.jpg
```

---
