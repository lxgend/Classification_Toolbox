# Classification_Toolbox

TNEWS

| Model             | Text representation   |    F1    |
| ----------------  | -------               |--------- |
| **FastText clf**  | 自训练FastText         |   0.63   | 
| **lgbm**          | 自训练FastText_sg      |   0.62   |
| **lgbm**          | tfidf                 |   0.55   |
| **lgbm**          | 腾讯WV                 |   0.63   |
| **SVM**           | 腾讯WV                 |   0.61   |

## Todo
- [x] BERT
- [ ] XLNet
- [x] FastText
- [x] lgbm
- [ ] TextCNN
- [ ] DPCNN
- [ ] GCN
- [ ] ...

## Reference
### data
* [TNEWS' 今日头条中文新闻，短文本15分类](https://github.com/CLUEbenchmark/CLUE)

### article
* [机器学习大乱斗](https://github.com/wavewangyue/text-classification)
* [阿里云tianchi新闻文本分类大赛rank4分享](https://mp.weixin.qq.com/s/ceT0Cu2KF7yQVSsXDYezXQ)