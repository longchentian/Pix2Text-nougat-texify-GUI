# Pix2Text-nougat-texify-GUI-offline

## 工程来源

感谢这些github上的工程：

[LaTeX-OCR: pix2tex](https://github.com/lukas-blecher/LaTeX-OCR)提供GUI；

[Pix2Text](https://github.com/breezedeus/Pix2Text/tree/main)提供模型Model；

[nougat-latex-ocr](https://github.com/NormXU/nougat-latex-ocr)提供模型Model；

[texify](https://github.com/VikParuchuri/texify) 提供模型、提供streamlit_app代码。



## GUI

### Pix2Text

![image-20240312092614970](./README.assets/image-20240312092614970.png)

### Nougat_Latex

![image-20240312101032252](./README.assets/image-20240312101032252.png)

### Texify

![image-20240312103203354](./README.assets/image-20240312103203354.png)

### streamlit power by Texify

![image-20240312131850345](./README.assets/image-20240312131850345.png)

## 后记

才疏学浅，代码写的很Low,不会Qt,只会简单调用模型。

读了下Qt的代码，大概会用Qt了，所以采用做了个改进版本。详见Master分支
主要添加功能
- [x] 添加了LaTex转MathML
- [x] 添加了监听剪切板
详细见：
[Pix2Text-GUI](https://github.com/longchentian/Pix2Text-nougat-texify-GUI/tree/Master)

