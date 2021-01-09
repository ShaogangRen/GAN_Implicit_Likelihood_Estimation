# GAN Implicit likelihood estimation

pytorch demo  implementation of paper [Estimate the Implicit Likelihoods of GANs with Application to Anomaly Detection, WWW2020]

## Citation
If it is useful for your research or work, please cite our paper
```
@inproceedings{ren2020estimate,
  title={Estimate the Implicit Likelihoods of GANs with Application to Anomaly Detection},
  author={Ren, Shaogang and Li, Dingcheng and Zhou, Zhixin and Li, Ping},
  booktitle={Proceedings of The Web Conference 2020},
  pages={2287--2297},
  year={2020}
}
```

## Simulation data
step1:
```
cd model_simulation/
```

step2:
```
python3 main_sim.py
```

step3:
```
python3 main_sim.py --batch_size=50 --z_dim=2 --lrD=0.000004 --lrG=0.000004 --lrIG=0.000004 --input_size=6 --iter_gan=30000 --gpu_ids=0
```

Or

```
nohup python3 main_sim.py --batch_size=50 --z_dim=2 --lrD=0.000004 --lrG=0.000004 --lrIG=0.000004 --input_size=6 --iter_gan=30000 --gpu_ids=0 &
```


## Arrhythmia data
step1:
```
cd model_arrhythmia/
```

step2:
Downlowd ALAD(https://github.com/houssamzenati/Adversarially-Learned-Anomaly-Detection), unzip and change the folder name to 'ALAD'.


step3:
```
python3 main_arrhythmia.py --batch_size=30 --z_dim=50 --h_dim=128 --gpu_ids=0  --input_size=274 --iter_gan=200000 --llk_way=eig --dataset=arrhythmia --lrD=0.000004 --lrG=0.0000004 --lrIG=0.0000004 --out_dir=output
```

Or
```
nohup python3 main_arrhythmia.py --batch_size=30 --z_dim=50 --h_dim=128 --gpu_ids=0  --input_size=274 --iter_gan=200000 --llk_way=eig --dataset=arrhythmia --lrD=0.000004 --lrG=0.0000004 --lrIG=0.0000004 --out_dir=output  &
```

